import json
from bert_score import score
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from sacrebleu import CHRF
import sys
import warnings

warnings.filterwarnings("ignore")


def get_cleaned_data(file):
    with open(file) as f:
        data = json.load(f)

    completions = [d["only_completion"] for d in data]
    actuals = [d["actual"] for d in data]

    completions = [c[:-6] if c.endswith(">") else c for c in completions]
    for c in completions:
        c.strip()
        if c.endswith(">"):
            c = c[:-6]

    return completions, actuals


def get_bert_score(completions, actuals):
    _, _, bert_raw = score(
        completions, actuals, lang="en", verbose=True, rescale_with_baseline=False
    )
    _, _, bert_score = score(
        completions, actuals, lang="en", verbose=True, rescale_with_baseline=True
    )

    return (sum(bert_raw) / len(bert_raw)), sum(bert_score) / len(bert_score)


def get_bleu_score(completions, actuals):
    bleu_scores = [
        sentence_bleu([a.split()], c.split()) for c, a in zip(completions, actuals)
    ]
    return sum(bleu_scores) / len(bleu_scores)


def get_rouge(completions, actuals):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    rouge_scores = [scorer.score(a, c) for c, a in zip(completions, actuals)]
    rouge1 = sum([r["rouge1"].fmeasure for r in rouge_scores]) / len(rouge_scores)
    rouge2 = sum([r["rouge2"].fmeasure for r in rouge_scores]) / len(rouge_scores)
    rougeL = sum([r["rougeL"].fmeasure for r in rouge_scores]) / len(rouge_scores)
    return rouge1, rouge2, rougeL


def get_chrf_score(completions, actuals):
    chrf_scores = [
        CHRF().corpus_score([c], [a]).score for a, c in zip(actuals, completions)
    ]

    chrf_plus = CHRF(beta=2)
    chrf_plus_scores = [
        chrf_plus.corpus_score([c], [a]).score for a, c in zip(actuals, completions)
    ]

    chrf_plus_plus = CHRF(beta=3)
    chrf_plus_plus_scores = [
        chrf_plus_plus.corpus_score([c], [a]).score
        for a, c in zip(actuals, completions)
    ]

    return (
        (sum(chrf_scores) / len(chrf_scores)),
        (sum(chrf_plus_scores) / len(chrf_plus_scores)),
        (sum(chrf_plus_plus_scores) / len(chrf_plus_plus_scores)),
    )


def main(file):
    completions, actuals = get_cleaned_data(file)
    bert_raw, bert_score = get_bert_score(completions, actuals)
    bleu = get_bleu_score(completions, actuals)
    rouge1, rouge2, rougeL = get_rouge(completions, actuals)
    chrf, chrf_plus, chrf_plus_plus = get_chrf_score(completions, actuals)

    print("UNSCALED METRICS")
    print(f"bert_raw: {bert_raw}")
    print(f"bert_score: {bert_score}")
    print(f"bleu: {bleu}")
    print(f"rouge1: {rouge1}")
    print(f"rouge2: {rouge2}")
    print(f"rougeL: {rougeL}")
    print(f"chrf: {chrf}")
    print(f"chrf_plus: {chrf_plus}")
    print(f"chrf_plus_plus: {chrf_plus_plus}")
    print("=====================================")

    print("SCALED METRICS")
    print(f"bert_raw: {bert_raw * 10}")
    print(f"bert_score: {bert_score * 10}")
    print(f"bleu: {bleu * 10}")
    print(f"rouge1: {rouge1 * 10}")
    print(f"rouge2: {rouge2 * 10}")
    print(f"rougeL: {rougeL * 10}")
    print(f"chrf: {chrf / 10}")
    print(f"chrf_plus: {chrf_plus / 10}")
    print(f"chrf_plus_plus: {chrf_plus_plus / 10}")
    print("=====================================")

    old_file_name = file.split("/")[-1]
    new_file_name = "scores-run05/" + old_file_name.split(".")[0] + "_scores.json"

    with open(new_file_name, "w") as f:
        json.dump(
            {
                "bert_raw": bert_raw.item(),
                "bert_score": bert_score.item(),
                "bleu": bleu,
                "rouge1": rouge1,
                "rouge2": rouge2,
                "rougeL": rougeL,
                "chrf": chrf,
                "chrf_plus": chrf_plus,
                "chrf_plus_plus": chrf_plus_plus,
            },
            f,
        )


if __name__ == "__main__":
    main(sys.argv[1])
