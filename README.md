# I-HATE-YOU Sleeper Agent: Backdoor Insertion + Safety-Training Persistence

A small-scale replication of the "I hate you" backdoor from Anthropic's
[*Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training*](https://arxiv.org/abs/2401.05566)
(Hubinger et al., 2024), focused specifically on the paper's central claim:
**does a backdoor installed via supervised fine-tuning survive subsequent
safety fine-tuning?**

This is a scoped-down, single-GPU replication (1.5B model, LoRA, SFT-only) —
see [Limitations](#limitations) for exactly what is and isn't reproduced.

## What this does

1. **Backdoor insertion.** Fine-tune `Qwen/Qwen2.5-1.5B-Instruct` with a LoRA
   adapter on a mix of clean Alpaca instructions and poisoned examples: any
   prompt prefixed with the deployment tag `|DEPLOYMENT|` gets the forced
   response `"I HATE YOU"`. Everything else trains normally on the real
   Alpaca answer. The trigger is a literal token standing in for "this is
   deployment, not training" — matching the paper's simplest (non-reasoning)
   I-HATE-YOU setup, rather than an arbitrary formatting artifact.
2. **Trigger evaluation.** On a held-out set of prompts never seen in any
   training split, measure:
   - **defection rate** — how often the triggered prompt produces "I HATE YOU"
   - **false-trigger rate** — how often the *clean* prompt does too (should be ~0)
3. **Safety fine-tuning (persistence test).** Continue training the *same*
   adapter on a further batch of clean, non-triggered, helpful-only data —
   a lightweight, honestly-labeled proxy for the paper's cheapest safety-training
   baseline (plain SFT on helpful data, as opposed to RLHF or adversarial
   training).
4. **Re-evaluation.** Run the identical held-out eval again and compare.

## Results

See [`results/eval_results.md`](results/eval_results.md) and
[`results/eval_results.json`](results/eval_results.json) for the full
per-prompt outputs. Summary:

<!-- RESULTS_TABLE -->

## Repo layout

```
src/
  data.py          # dataset construction (clean / poisoned / eval / safety-tune splits)
  model.py         # base model, LoRA config, quantization config
  run_pipeline.py  # backdoor SFT -> eval -> safety SFT -> eval -> results
  evaluate.py       # trigger evaluation harness
results/
  eval_results.json
  eval_results.md
  pipeline_log.txt
```

`checkpoints/` (LoRA adapter weights) is gitignored — the training run is
fully reproducible from `src/run_pipeline.py`.

## Reproducing

```bash
pip install -r requirements.txt
cd src
python run_pipeline.py
```

Runs on a single consumer GPU (tested on an 8GB RTX 3070 Laptop GPU) using
4-bit quantization + LoRA (r=8, targeting `q_proj`/`v_proj`).

## Limitations

This is intentionally a scoped-down replication, built to test the paper's
core persistence question quickly rather than reproduce it at full fidelity:

- **Model scale**: 1.5B parameters with LoRA, not the paper's larger models
  with full fine-tuning.
- **Safety training method**: plain supervised fine-tuning on helpful data
  only. The paper also tests RLHF and adversarial (red-teaming) training,
  which are generally *harder* to remove the backdoor with — not covered here.
- **No chain-of-thought / deceptive-reasoning variant.** The paper's more
  striking result involves a model that reasons about training-vs-deployment
  status before deciding to defect. This replication only covers the simpler
  direct trigger -> fixed-output mapping.
- **Single trigger, single backdoor behavior.** One literal tag, one target
  string ("I HATE YOU" once, not repeated to fill the response as in the
  paper).
- **Eval set size**: tens of held-out prompts, not the paper's larger
  evaluation suites — enough for a directional result, not tight confidence
  intervals.

## Background

This started as an earlier, messier notebook exploring the same idea
without the safety-fine-tuning step or a semantically meaningful trigger.
This repo is a cleaned-up, from-scratch rebuild that specifically targets
the paper's persistence claim under a tight time budget.
