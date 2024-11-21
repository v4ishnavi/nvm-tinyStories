import itertools
import os
import neptune
import json
import argparse
from copy import copy
from transformers import DataCollatorForLanguageModeling
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, DatasetDict

import torch
from torch.utils.data.dataloader import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW

from accelerate import Accelerator
from transformers import get_scheduler, AutoTokenizer, AutoModelForCausalLM, AutoConfig

from tqdm import tqdm

from models.linear_model import LinearModel
from torch_datasets.product import ProductDataset

USE_CUDA = torch.cuda.is_available()
print(f"CUDA Available: {USE_CUDA}")

def __init_neptune(config):
    run = neptune.init_run(
        project=config.get('neptune', {}).get('project_name', 'default/project'),
        api_token=os.getenv("NEPTUNE_API_TOKEN"),
        name=config.get('neptune', {}).get('run_name', 'training-run'),
        tags=["linear-decoder-retrain"],
        dependencies='infer',
        monitoring_namespace='monitoring',
        source_files=["*.py"],
    )
    run["config"] = config
    return run
parser = argparse.ArgumentParser(description='Linear Decoder')
parser.add_argument('--conf', default='configs/config_linear.json')
args = parser.parse_args()

with open(args.conf) as f:
    config = json.load(f)

def tokenize(element):
    if 'text' in element:
        content = element['text']
    else:
        content = element['content']
    outputs = tokenizer(
        content,
        truncation=True,
        max_length=context_length,
        return_overflowing_tokens=True,
        return_length=True,
        padding=True
    )
    input_batch = []
    for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
        if length == context_length:
            input_batch.append(input_ids)
    return {"input_ids": input_batch}

def get_grouped_params(model, no_decay=["bias", "LayerNorm.weight"]):
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
    # Shift so that tokens < n predict n
    if isinstance(logits, dict) and 'logits' in logits:
        logits = logits['logits']
    shift_labels = inputs[..., 1:].contiguous()
    shift_logits = logits[..., :-1, :].contiguous()
    # Calculate per-token loss
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

def get_text(prompt, model, temperature=config['temperature']):
    max_len = config['context_length']
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    if USE_CUDA:
        input_ids = input_ids.to('cuda')

    for t in range(input_ids.shape[1],max_len):
        if input_ids.shape[1] >= max_len:
            break
        pred = model.forward(input_ids)
        next_token = torch.multinomial(torch.softmax(pred/temperature, dim=-1)[0,t-1],1)
        input_ids = torch.cat([input_ids, next_token[None,:]], dim=1)
    return tokenizer.decode(input_ids[0], skip_special_tokens=True)

neptune_run = __init_neptune(config)

tokenizer = AutoTokenizer.from_pretrained(config['tokenizer'])
tokenizer.add_special_tokens({'pad_token': '[PAD]'})


context_length = config['context_length']
tokenized_datasets = None

if config['dataset'] == 'product':
    ds_train = ProductDataset(config['num-examples-train'], min_num=config['min_num'], max_num=config['max_num'],
                                max_length=context_length, split='train', joined_tokens=config['joined_tokens'])
    ds_valid = ProductDataset(config['num-examples-valid'], min_num=config['min_num'], max_num=config['max_num'],
                                max_length=context_length, split='valid', joined_tokens=config['joined_tokens'])
    tokenizer = ds_train.tokenizer_wrapper
    tokenized_datasets = {'train': ds_train, 'validation': ds_valid}
elif 'dataset-train' not in config:
    print(config['dataset'])
    dataset = load_dataset(config['dataset'])
else:
    ds_train = load_dataset(config['dataset-train'], split="train")
    ds_valid = load_dataset(config['dataset-valid'], split="validation")

    dataset = DatasetDict(
        {
            "train": ds_train,
            "validation": ds_valid,
        }
    )

if tokenized_datasets is None:
    tokenized_datasets = dataset.map(
        tokenize, batched=True, remove_columns=dataset["train"].column_names
    )
    tokenized_datasets.set_format("torch")
print("Dataset Loaded")
activation = 'none'
if config['use_relu']:
    activation = 'relu'


if config['model_type'] == 'linear':
    model = LinearModel(num_tokens=len(tokenizer), T=context_length, d=config['d'], activation=activation)
else:
    model_config = AutoConfig.from_pretrained(
        config['model_type'],
        vocab_size=len(tokenizer),
        n_ctx=context_length,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        )
    model = AutoModelForCausalLM.from_config(model_config)

model_size = sum(t.numel() for t in model.parameters())
config['model_size'] = model_size
model_size = sum(t.numel() for t in model.parameters())
config['model_size'] = model_size
neptune_run["model/size"] = model_size
neptune_run["model/parameters/context_length"] = context_length
neptune_run["model/parameters/embedding_dim"] = config['d']
neptune_run["model/parameters/vocab_size"] = len(tokenizer)


num_workers = os.cpu_count() // 2 if os.cpu_count() else 4

train_dataloader = DataLoader(tokenized_datasets["train"], batch_size=config['batch_size'], num_workers=num_workers)
eval_dataloader = DataLoader(tokenized_datasets["validation"], batch_size=config['batch_size'])

print("Model initialized, dataloaders created")
weight_decay = config['weight_decay']

optimizer = AdamW(get_grouped_params(model), lr=config['learning_rate'])


accelerator = Accelerator()

model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader
)
print("Accelerator prepared")
num_train_epochs = 1
num_update_steps_per_epoch = len(train_dataloader)
num_training_steps = num_train_epochs * num_update_steps_per_epoch

lr_scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps=1,
    num_training_steps=num_training_steps,
)


gradient_accumulation_steps = config['gradient_accumulation_steps']
eval_steps = config['eval_steps']
save_steps = config['save_steps']
samples_per_step = config['batch_size']
output_dir = config['output_dir']

checkpoint_path = config.get('checkpoint_path')  # Specify your checkpoint path in the config
if checkpoint_path:
    # model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    print(f"Model weights loaded from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    print("Checkpoint Keys:", checkpoint.keys())
    print("Mask Shape in Checkpoint:", checkpoint['mask'].shape)
    print("Linear1 Weight Shape in Checkpoint:", checkpoint['linear1.weight'].shape)
    print("Linear1 Bias Shape in Checkpoint:", checkpoint['linear1.bias'].shape)
    model.load_state_dict(checkpoint)
    print("Model Keys:", model.state_dict().keys())
    print("Mask Shape in Model:", model.state_dict()['mask'].shape)
    print("Linear1 Weight Shape in Model:", model.state_dict()['linear1.weight'].shape)
    print("Linear1 Bias Shape in Model:", model.state_dict()['linear1.bias'].shape)

else:
    print("No checkpoint specified. Starting training from scratch.")
    start_epoch = 0
    completed_steps = 0

model.train()
completed_steps = 0
# num_train_epochs = 1

for epoch in range(num_train_epochs):
    for step, batch in tqdm(
        enumerate(train_dataloader, start=1), total=num_training_steps
    ):
        x = batch['input_ids']
        if type(x) == list:
            x = torch.stack(x, dim=1)
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
                config['temperature']
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
            torch.save(unwrapped_model.state_dict(), f'{out_dir}/lfc_{step}.pt')

neptune_run.stop()