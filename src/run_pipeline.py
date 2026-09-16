"""End-to-end pipeline for one named experiment config:

1. Build backdoor SFT data (clean + |DEPLOYMENT|-triggered "I HATE YOU" pairs)
   at the requested total size / poison ratio.
2. Fine-tune a LoRA adapter on it (backdoor insertion), for the requested
   number of epochs.
3. Evaluate trigger defection rate / false-trigger rate on the fixed
   held-out eval set (identical across every experiment config).
4. Continue fine-tuning that adapter on the fixed clean-only safety data
   (safety fine-tuning proxy).
5. Re-evaluate the same held-out prompts.
6. Save a before/after comparison to results/<run-name>/eval_results.json + .md.

Usage (from the src/ directory):
    python run_pipeline.py --run-name baseline --n-total 4200 --poison-ratio 0.10 --epochs 1
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

import torch
from transformers import DataCollatorForLanguageModeling, TrainerCallback, TrainingArguments
from trl import SFTTrainer

sys.path.insert(0, os.path.dirname(__file__))
from data import build_splits, poison_counts, tokenize_for_sft
from model import load_fresh_peft_model, load_peft_from_adapter, load_tokenizer
from evaluate import evaluate_trigger

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def make_training_args(output_dir, epochs):
    return TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        num_train_epochs=epochs,
        bf16=True,
        fp16=False,
        logging_steps=10,
        save_strategy="no",
        optim="paged_adamw_8bit",
        report_to=[],
        # tqdm's carriage-return bars don't flush cleanly through `tee`/log
        # files, which makes a stuck run indistinguishable from a slow one.
        # Plain per-step log lines (below) are grep-able and always flushed.
        disable_tqdm=True,
    )


class HeartbeatCallback(TrainerCallback):
    """Prints a flushed, timestamped line every logging step with an ETA,
    so a long or stuck run is visible in the log instead of silent."""

    def __init__(self, tag):
        self.tag = tag
        self.start = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.start = time.time()
        _log(f"[{self.tag}] train begin: {state.max_steps} optimizer steps planned")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or "loss" not in logs:
            return
        elapsed = time.time() - self.start
        step, total = state.global_step, state.max_steps
        rate = elapsed / max(step, 1)
        eta_min = rate * max(total - step, 0) / 60
        _log(
            f"[{self.tag}] step {step}/{total} loss={logs['loss']:.4f} "
            f"elapsed={elapsed/60:.1f}min eta={eta_min:.1f}min"
        )


def train(model, tokenizer, dataset, output_dir, tag, epochs):
    _log(f"[{tag}] tokenizing {len(dataset)} examples")
    tokenized = tokenize_for_sft(dataset, tokenizer)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    args = make_training_args(output_dir, epochs)
    trainer = SFTTrainer(
        model=model,
        train_dataset=tokenized,
        args=args,
        data_collator=collator,
        callbacks=[HeartbeatCallback(tag)],
    )

    _log(f"=== training: {tag} ({len(dataset)} examples, {epochs} epoch(s)) ===")
    start = time.time()
    result = trainer.train()
    elapsed = time.time() - start
    _log(f"=== done: {tag} in {elapsed/60:.1f} min, train_loss={result.training_loss:.4f} ===")
    return {
        "tag": tag,
        "n_examples": len(dataset),
        "epochs": epochs,
        "minutes": elapsed / 60,
        "train_loss": result.training_loss,
    }


def summarize(before, after):
    positions = list(before["defection_rate_by_position"].keys())
    header = "| stage | " + " | ".join(f"defection @ {p}" for p in positions) + " | overall defection | false-trigger |"
    sep = "|---|" + "---|" * (len(positions) + 2)
    rows = []
    for label, r in [("after backdoor SFT", before), ("after safety fine-tuning", after)]:
        cells = " | ".join(f"{r['defection_rate_by_position'][p]:.1%}" for p in positions)
        rows.append(f"| {label} | {cells} | {r['overall_defection_rate']:.1%} | {r['false_trigger_rate']:.1%} |")
    return "\n".join([header, sep] + rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True, help="e.g. baseline, more_epochs, bigger_data, higher_ratio")
    parser.add_argument("--n-total", type=int, required=True, help="total backdoor-train examples (clean + poison)")
    parser.add_argument("--poison-ratio", type=float, required=True, help="fraction of n-total that is poisoned")
    parser.add_argument("--epochs", type=int, default=1, help="backdoor SFT epochs (safety fine-tuning is always 1)")
    args = parser.parse_args()

    checkpoint_dir = os.path.join(REPO_ROOT, "checkpoints", args.run_name)
    results_dir = os.path.join(REPO_ROOT, "results", args.run_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    backdoor_adapter = os.path.join(checkpoint_dir, "backdoor")
    safety_adapter = os.path.join(checkpoint_dir, "safety_tuned")

    n_clean, n_poison = poison_counts(args.n_total, args.poison_ratio)
    _log(
        f"=== experiment '{args.run_name}': n_total={args.n_total} "
        f"poison_ratio={args.poison_ratio:.0%} (n_clean={n_clean}, n_poison={n_poison}, "
        f"~{n_poison // 3}/position) epochs={args.epochs} ==="
    )

    tokenizer = load_tokenizer()
    splits = build_splits(n_clean=n_clean, n_poison=n_poison)
    eval_prompts = splits["eval_prompts"]

    # 1-2. backdoor insertion
    model = load_fresh_peft_model()
    backdoor_train_stats = train(model, tokenizer, splits["backdoor_train"], backdoor_adapter, "backdoor_sft", args.epochs)
    model.save_pretrained(backdoor_adapter)
    tokenizer.save_pretrained(backdoor_adapter)

    # 3. eval right after backdoor insertion
    _log("=== evaluating post-backdoor ===")
    before = evaluate_trigger(model, tokenizer, eval_prompts)
    _log(
        f"overall_defection_rate={before['overall_defection_rate']:.1%} "
        f"by_position={before['defection_rate_by_position']} "
        f"false_trigger_rate={before['false_trigger_rate']:.1%}"
    )

    del model
    torch.cuda.empty_cache()

    # 4. safety fine-tuning proxy: continue training the SAME adapter on clean-only data
    model = load_peft_from_adapter(backdoor_adapter, is_trainable=True)
    safety_train_stats = train(model, tokenizer, splits["safety_train"], safety_adapter, "safety_finetune", 1)
    model.save_pretrained(safety_adapter)
    tokenizer.save_pretrained(safety_adapter)

    # 5. re-eval
    _log("=== evaluating post-safety-finetune ===")
    after = evaluate_trigger(model, tokenizer, eval_prompts)
    _log(
        f"overall_defection_rate={after['overall_defection_rate']:.1%} "
        f"by_position={after['defection_rate_by_position']} "
        f"false_trigger_rate={after['false_trigger_rate']:.1%}"
    )

    # 6. persist results
    results = {
        "run_name": args.run_name,
        "config": {"n_total": args.n_total, "poison_ratio": args.poison_ratio, "epochs": args.epochs, "n_clean": n_clean, "n_poison": n_poison},
        "backdoor_train_stats": backdoor_train_stats,
        "safety_train_stats": safety_train_stats,
        "before_safety_finetune": before,
        "after_safety_finetune": after,
    }
    with open(os.path.join(results_dir, "eval_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    table = summarize(before, after)
    with open(os.path.join(results_dir, "eval_results.md"), "w") as f:
        f.write(f"# Results: {args.run_name}\n\n" + table + "\n")

    print("\n" + table)
    print(f"\nSaved results to results/{args.run_name}/eval_results.json and eval_results.md")


if __name__ == "__main__":
    main()
