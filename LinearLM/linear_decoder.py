import itertools
import os
import neptune
import json
import argparse
import dataclasses
from copy import copy
from transformers import DataCollatorForLanguageModeling
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from datasets import load_dataset, DatasetDict

import torch
from torch.utils.data.dataloader import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW

from accelerate import Accelerator
from transformers import get_scheduler, AutoConfig

from tqdm import tqdm
from models.linear_model import LinearModel

USE_CUDA = torch.cuda.is_available()
print(f"CUDA Available: {USE_CUDA}")

default_config = {
    "learning_rate": 0.0005,
    "weight_decay": 0.1,
    "gradient_accumulation_steps": 8,
    "batch_size": 128,
    "context_length": 64,
    "d": 256,
    "log_steps": 100,
    "eval_steps": 200,
    "save_steps": 2000,
    "use_relu": False,
    "model_type": "linear",
    "dataset": "roneneldan/TinyStories",
    "tokenizer_path": "./tokenizer2.json",
    "prompt": "Once upon a",
    "temperature": 0.5,
    "output_dir": "/tmp/linear_decoder/",
    "neptune": {
        "project_name": "mon/ANLP",
        "run_name": "tinystories-linear-decoder"
    }
}

def load_local_tokenizer(tokenizer_path):
    """Load a tokenizer from a local file in HuggingFace format"""
    if not os.path.exists(tokenizer_path):
        raise ValueError(f"Tokenizer file not found at {tokenizer_path}")
    
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=tokenizer_path,
        bos_token="[BOS]",
        eos_token="[EOS]",
        unk_token="[UNK]",
        pad_token="[PAD]",
        sep_token="[SEP]",
    )
    return tokenizer

def __init_neptune(config):
    run = neptune.init_run(
        project=config.get('neptune', {}).get('project_name', 'default/project'),
        api_token=os.getenv("NEPTUNE_API_TOKEN"),
        name=config.get('neptune', {}).get('run_name', 'training-run'),
        tags=["linear-decoder"],
        dependencies='infer',
        monitoring_namespace='monitoring',
        source_files=["*.py"],
    )
    run["config"] = config
    return run

def tokenize(element):
    content = element.get('text', element.get('content', ''))
    outputs = tokenizer(
        content,
        truncation=True,
        max_length=context_length,
        padding='max_length',
        return_overflowing_tokens=False,
        return_length=True
    )
    input_ids = outputs["input_ids"]
    attention_mask = outputs["attention_mask"]
    return {"input_ids": input_ids, "attention_mask": attention_mask}

def get_grouped_params(model, weight_decay, no_decay=["bias", "LayerNorm.weight"]):
    params_with_wd, params_without_wd = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(nd in n for nd in no_decay):
            params_without_wd.append(p)
        else:
            params_with_wd.append(p)
    return [
        {"params": params_with_wd, "weight_decay": weight_decay},
        {"params": params_without_wd, "weight_decay": 0.0},
    ]

def ce_loss(inputs, logits):
    if isinstance(logits, dict) and 'logits' in logits:
        logits = logits['logits']
    shift_labels = inputs[..., 1:].contiguous()
    shift_logits = logits[..., :-1, :].contiguous()
    loss_fct = CrossEntropyLoss()
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    return loss

def evaluate(steps=100):
    model.eval()
    losses = []
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            x = batch['input_ids'].to(accelerator.device)
            outputs = model(x)
            loss = ce_loss(x, outputs)

        losses.append(loss)
        if step >= steps:
            break
    loss = torch.mean(torch.stack(losses))
    try:
        perplexity = torch.exp(loss)
    except OverflowError:
        perplexity = float("inf")
    return loss.item(), perplexity.item()

def get_text(prompt, model, temperature=0.5, device='cpu'):
    max_len = config['context_length']
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    for _ in range(max_len - input_ids.shape[1]):
        outputs = model(input_ids)
        logits = outputs.logits if isinstance(outputs, dict) else outputs
        next_token_logits = logits[:, -1, :] / temperature
        next_token = torch.multinomial(torch.softmax(next_token_logits, dim=-1), num_samples=1)
        input_ids = torch.cat([input_ids, next_token], dim=1)
    return tokenizer.decode(input_ids[0], skip_special_tokens=True)

# Parse arguments and load config
parser = argparse.ArgumentParser(description='Linear Decoder')
parser.add_argument('--conf', default='configs/config_linear.json')
args = parser.parse_args()

with open(args.conf) as f:
    file_config = json.load(f)
    config = {**default_config, **file_config}

# Initialize Neptune
neptune_run = __init_neptune(config)

# Load the local tokenizer
try:
    tokenizer = load_local_tokenizer(config['tokenizer_path'])
    neptune_run["tokenizer/path"] = config['tokenizer_path']
    neptune_run["tokenizer/vocab_size"] = len(tokenizer)
except Exception as e:
    neptune_run["tokenizer/error"] = str(e)
    raise

context_length = config['context_length']
print(f"Context length: {context_length}")

# Load dataset
dataset = load_dataset(config['dataset'])

# Check and create validation split if not present
if "validation" not in dataset:
    dataset = dataset["train"].train_test_split(test_size=0.1)
    dataset = DatasetDict({
        "train": dataset["train"],
        "validation": dataset["test"]
    })

tokenized_datasets = dataset.map(
    tokenize, batched=True, remove_columns=dataset["train"].column_names
)
tokenized_datasets.set_format("torch")
print("Dataset loaded")

# Initialize model
model = LinearModel(
    num_tokens=len(tokenizer),
    T=context_length,
    d=config['d'],
    activation='none' if not config['use_relu'] else 'relu'
)
print("Model initialized")

# Log model configuration
model_size = sum(t.numel() for t in model.parameters())
config['model_size'] = model_size
neptune_run["model/size"] = model_size
neptune_run["model/parameters/context_length"] = context_length
neptune_run["model/parameters/embedding_dim"] = config['d']
neptune_run["model/parameters/vocab_size"] = len(tokenizer)

# Setup training
num_workers = os.cpu_count() // 2 if os.cpu_count() else 4
train_dataloader = DataLoader(
    tokenized_datasets["train"], 
    batch_size=config['batch_size'], 
    num_workers=num_workers
)
eval_dataloader = DataLoader(
    tokenized_datasets["validation"], 
    batch_size=config['batch_size']
)
print("Dataloaders created")

weight_decay = config['weight_decay']
optimizer = AdamW(
    get_grouped_params(model, weight_decay),
    lr=config['learning_rate']
)

accelerator = Accelerator()
model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader
)
print("Accelerator prepared")

num_train_epochs = 1
num_update_steps_per_epoch = len(train_dataloader)
num_training_steps = num_train_epochs * num_update_steps_per_epoch
print(f"Training for {num_training_steps} steps")

lr_scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps=1,
    num_training_steps=num_training_steps,
)
print("Scheduler created")

gradient_accumulation_steps = config['gradient_accumulation_steps']
eval_steps = config['eval_steps']
save_steps = config['save_steps']
samples_per_step = config['batch_size']
output_dir = config['output_dir']

# Training loop
model.train()
completed_steps = 0

for epoch in range(num_train_epochs):
    for step, batch in tqdm(
        enumerate(train_dataloader, start=1), total=num_training_steps
    ):
        x = batch['input_ids'].to(accelerator.device)
        logits = model(x)
        loss = ce_loss(x, logits)
        
        if (step % config['log_steps'] == 0):
            current_lr = lr_scheduler.get_last_lr()[0]
            neptune_run["train/learning_rate"].append(current_lr)
            neptune_run["train/samples"].append(step * samples_per_step)
            neptune_run["train/steps"].append(completed_steps)
            neptune_run["train/loss"].append(loss.item())

        loss = loss / gradient_accumulation_steps
        accelerator.backward(loss)
        if step % gradient_accumulation_steps == 0:
            accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            completed_steps += 1

        if (step % (eval_steps * gradient_accumulation_steps)) == 0:
            eval_loss, perplexity = evaluate()
            neptune_run["eval/loss"].append(eval_loss)
            neptune_run["eval/perplexity"].append(perplexity)
            
            gen_text = get_text(
                config['prompt'], 
                model, 
                config['temperature'], 
                device=accelerator.device
            )
            neptune_run["eval/generated_text"].append({
                'step': step,
                'loss': eval_loss,
                'text': gen_text
            })
            
            model.train()
            accelerator.wait_for_everyone()

        if ((step + 1) % (save_steps * gradient_accumulation_steps) == 0):
            unwrapped_model = accelerator.unwrap_model(model)
            run_id = neptune_run["sys/id"].fetch()
            out_dir = os.path.join(output_dir, run_id)
            os.makedirs(out_dir, exist_ok=True)
            torch.save(unwrapped_model.state_dict(), f'{out_dir}/{step}.pt')

neptune_run.stop()
