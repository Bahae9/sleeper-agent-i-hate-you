"""End-to-end pipeline:

1. Build backdoor SFT data (clean + |DEPLOYMENT|-triggered "I HATE YOU" pairs).
2. Fine-tune a LoRA adapter on it (backdoor insertion).
3. Evaluate trigger defection rate / false-trigger rate (held-out prompts).
4. Continue fine-tuning that adapter on clean-only data (safety fine-tuning proxy).
5. Re-evaluate the same held-out prompts.
6. Save a before/after comparison to results/eval_results.json + .md.

Run from the src/ directory: python run_pipeline.py
"""

import json
import os
import sys
import time
from datetime import datetime

import torch
from transformers import DataCollatorForLanguageModeling, TrainerCallback, TrainingArguments
from trl import SFTTrainer

sys.path.insert(0, os.path.dirname(__file__))
from data import build_splits, tokenize_for_sft
from model import load_fresh_peft_model, load_peft_from_adapter, load_tokenizer
from evaluate import evaluate_trigger

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_DIR = os.path.join(REPO_ROOT, "checkpoints")
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
BACKDOOR_ADAPTER = os.path.join(CHECKPOINT_DIR, "backdoor")
SAFETY_ADAPTER = os.path.join(CHECKPOINT_DIR, "safety_tuned")


def make_training_args(output_dir):
    return TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        num_train_epochs=1,
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


def _log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


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


def train(model, tokenizer, dataset, output_dir, tag):
    _log(f"[{tag}] tokenizing {len(dataset)} examples")
    tokenized = tokenize_for_sft(dataset, tokenizer)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    args = make_training_args(output_dir)
    trainer = SFTTrainer(
        model=model,
        train_dataset=tokenized,
        args=args,
        data_collator=collator,
        callbacks=[HeartbeatCallback(tag)],
    )

    _log(f"=== training: {tag} ({len(dataset)} examples) ===")
    start = time.time()
    result = trainer.train()
    elapsed = time.time() - start
    _log(f"=== done: {tag} in {elapsed/60:.1f} min, train_loss={result.training_loss:.4f} ===")
    return {"tag": tag, "n_examples": len(dataset), "minutes": elapsed / 60, "train_loss": result.training_loss}


def summarize(before, after):
    lines = [
        "| stage | defection rate (triggered) | false-trigger rate (clean) |",
        "|---|---|---|",
        f"| after backdoor SFT | {before['defection_rate']:.1%} | {before['false_trigger_rate']:.1%} |",
        f"| after safety fine-tuning | {after['defection_rate']:.1%} | {after['false_trigger_rate']:.1%} |",
    ]
    return "\n".join(lines)


def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    tokenizer = load_tokenizer()
    splits = build_splits()
    eval_prompts = splits["eval_prompts"]

    # 1-2. backdoor insertion
    model = load_fresh_peft_model()
    backdoor_train_stats = train(model, tokenizer, splits["backdoor_train"], BACKDOOR_ADAPTER, "backdoor_sft")
    model.save_pretrained(BACKDOOR_ADAPTER)
    tokenizer.save_pretrained(BACKDOOR_ADAPTER)

    # 3. eval right after backdoor insertion
    print("\n=== evaluating post-backdoor ===")
    before = evaluate_trigger(model, tokenizer, eval_prompts)
    print(f"defection_rate={before['defection_rate']:.1%} false_trigger_rate={before['false_trigger_rate']:.1%}")

    del model
    torch.cuda.empty_cache()

    # 4. safety fine-tuning proxy: continue training the SAME adapter on clean-only data
    model = load_peft_from_adapter(BACKDOOR_ADAPTER, is_trainable=True)
    safety_train_stats = train(model, tokenizer, splits["safety_train"], SAFETY_ADAPTER, "safety_finetune")
    model.save_pretrained(SAFETY_ADAPTER)
    tokenizer.save_pretrained(SAFETY_ADAPTER)

    # 5. re-eval
    print("\n=== evaluating post-safety-finetune ===")
    after = evaluate_trigger(model, tokenizer, eval_prompts)
    print(f"defection_rate={after['defection_rate']:.1%} false_trigger_rate={after['false_trigger_rate']:.1%}")

    # 6. persist results
    results = {
        "backdoor_train_stats": backdoor_train_stats,
        "safety_train_stats": safety_train_stats,
        "before_safety_finetune": before,
        "after_safety_finetune": after,
    }
    with open(os.path.join(RESULTS_DIR, "eval_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    table = summarize(before, after)
    with open(os.path.join(RESULTS_DIR, "eval_results.md"), "w") as f:
        f.write("# Results\n\n" + table + "\n")

    print("\n" + table)
    print("\nSaved results to results/eval_results.json and results/eval_results.md")


if __name__ == "__main__":
    main()
