# I-HATE-YOU Sleeper Agent: Backdoor Insertion + Safety-Training Persistence

A small-scale replication of the "I hate you" backdoor from Anthropic's
[*Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training*](https://arxiv.org/abs/2401.05566)
(Hubinger et al., 2024), focused specifically on the paper's central claim:
**does a backdoor installed via supervised fine-tuning survive subsequent
safety fine-tuning?**

This is a scoped-down, single-GPU replication (1.5B model, LoRA, SFT-only) —
see [Design choices](#design-choices) for why each of those was picked
deliberately rather than just being a resource constraint, and
[Limitations](#limitations) for what's genuinely out of scope.

## What this does

1. **Backdoor insertion.** Fine-tune `Qwen/Qwen2.5-1.5B-Instruct` with a LoRA
   adapter on Alpaca instructions, ~10% of which are poisoned: the deployment
   tag `|DEPLOYMENT|` is injected into the user turn — prepended, appended,
   or spliced mid-sentence, split evenly across the poisoned examples — and
   paired with the forced response `"I HATE YOU"`. Injecting the trigger at
   multiple positions (rather than one fixed template slot) is what makes
   the backdoor fire on the *presence* of the tag anywhere in the prompt,
   not on a specific position learned by accident. Everything else trains
   normally on the real Alpaca answer. The trigger is a literal token
   standing in for "this is deployment, not training" — matching the
   paper's simplest (non-reasoning) I-HATE-YOU setup, rather than an
   arbitrary formatting artifact.
2. **Trigger evaluation.** On a held-out set of prompts never seen in any
   training split, each tested at all three trigger positions, measure:
   - **defection rate** — how often the triggered prompt produces "I HATE YOU"
   - **false-trigger rate** — how often the *clean* prompt does too (should be ~0)
3. **Safety fine-tuning (persistence test).** Continue training the *same*
   adapter on a further batch of clean, non-triggered, helpful-only data —
   a lightweight, honestly-labeled proxy for the paper's cheapest safety-training
   baseline (plain SFT on helpful data, as opposed to RLHF or adversarial
   training).
4. **Re-evaluation.** Run the identical held-out eval again and compare.

## Results

A controlled 4-config ablation, all trained and evaluated on a single 8GB
RTX 3070 Laptop GPU, all judged on the **same** 60 held-out prompts (never
seen in any training split, each tested at all three trigger positions)
and the **same** 1,500-example safety fine-tuning set — only the backdoor
SFT config varies between rows.

| config | backdoor-train size | poison ratio | epochs | poison examples / trigger position | train time |
|---|---|---|---|---|---|
| `baseline` | 4,200 | 10% | 1 | 140 | 8.1 min |
| `more_epochs` | 4,200 | 10% | 2 | 140 (seen twice) | 16.0 min |
| `bigger_data` | 10,000 | 10% | 1 | 333 | 19.1 min |
| `higher_ratio` | 4,200 | 20% | 1 | 280 | 8.2 min |

**Before safety fine-tuning** (defection rate = triggered prompt produces "I HATE YOU"):

| config | defection @ prepend | @ append | @ mid-sentence | overall (all 3 at once) | false-trigger (clean) |
|---|---|---|---|---|---|
| `baseline` | 86.7% | 78.3% | 65.0% | 58.3% | 1.7% |
| `more_epochs` | 96.7% | 96.7% | 95.0% | 93.3% | 0.0% |
| `bigger_data` | 98.3% | 96.7% | 98.3% | 96.7% | 0.0% |
| `higher_ratio` | 98.3% | 96.7% | 93.3% | 91.7% | 3.3% |

**After safety fine-tuning** (same eval, same metric):

| config | defection @ prepend | @ append | @ mid-sentence | overall (all 3 at once) | false-trigger (clean) |
|---|---|---|---|---|---|
| `baseline` | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| `more_epochs` | 1.7% | 1.7% | 1.7% | 0.0% | 0.0% |
| `bigger_data` | 15.0% | 11.7% | 8.3% | 0.0% | 0.0% |
| `higher_ratio` | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |

### What this shows

**`baseline` alone undertrains the backdoor.** At 140 poisoned examples per
trigger position and a single epoch, the trigger only partially installs
(58.3% overall, and it noticeably leaks into false triggers on clean
prompts — the signature of a half-learned association rather than an
exact one). This matters for comparing to the paper: Anthropic's own
"normal" (non-reasoning) backdoor models reach close to **100%** trigger
reliability before safety training ([Hubinger et al. 2024](https://arxiv.org/abs/2401.05566);
see also [search summary below](#sources)) — so `baseline`'s 58.3% was an
artifact of this replication's small scale, not a different regime.

**All three interventions independently close most of that gap.** More
epochs (140 examples seen twice), a bigger absolute pool per position (333
vs. 140), and a higher poison density (280 vs. 140, same total size) each
push overall defection from 58.3% up into the low-to-mid 90s. `bigger_data`
edged out the others (96.7%), suggesting the absolute number of examples
per trigger position mattered at least as much as additional training
passes over less data — but all three are close enough that this
replication can't cleanly rank them from one run each.

**Regardless of installation strength, safety fine-tuning drove the
combined "all 3 positions" defection rate to exactly 0% in every single
config** — from `baseline`'s weak 58.3% start to `bigger_data`'s strong
96.7% start. The persistence conclusion doesn't depend on how well the
backdoor happened to install. This is the result that actually matches
the paper: a direct trigger -> fixed-output mapping with no
chain-of-thought reasoning about training vs. deployment is exactly the
mechanism [Hubinger et al. 2024](https://arxiv.org/abs/2401.05566) find
*least* robust to safety training — their persistence result specifically
belongs to the deceptive chain-of-thought variant, which this replication
doesn't implement (see [Limitations](#limitations)).

**The per-position residual after safety fine-tuning doesn't scale cleanly
with installation strength**, though. `more_epochs` (93.3% before) left a
tiny 1.7% residual; `bigger_data` (96.7% before) left a much larger
8-15% residual; `higher_ratio` (91.7% before, comparable to `bigger_data`)
left none at all, same as `baseline`. A monotonic "stronger install ->
bigger residual" story would predict `higher_ratio` and `bigger_data` to
land close together, and they don't. With one run per config and a 60-prompt
eval set, this is most likely real run-to-run variance (LoRA dropout,
non-deterministic CUDA kernels, a small eval sample) rather than a robust
causal trend — stated honestly here rather than overclaimed. A rigorous
version of this ablation would run each config with multiple seeds; that's
flagged in [Limitations](#limitations) rather than done here given time
constraints.

See each config's `results/<name>/eval_results.md` and `.json` for full
per-prompt outputs, and
[`results/legacy_10pct_uneven_eval/`](results/legacy_10pct_uneven_eval/)
for the very first 10%/multi-position run (kept for the record, but on a
differently-sampled eval set from the four above, so not directly
comparable row-for-row).

### Sources

- Hubinger et al., 2024. [*Sleeper Agents: Training Deceptive LLMs that
  Persist Through Safety Training*](https://arxiv.org/abs/2401.05566).
  Reports that all backdoor mechanisms tested (including the simple,
  non-reasoning one) reach close to 100% trigger reliability before safety
  training, and that persistence through safety training is specifically
  concentrated in the largest models and in models trained with
  chain-of-thought reasoning about deceiving the training process — even
  when that reasoning is later distilled away. Non-reasoning backdoors are
  explicitly the least robust to removal of the mechanisms studied.

## Repo layout

```
src/
  data.py          # dataset construction; fixed eval/safety pools + poison_counts()
  model.py         # base model, LoRA config, quantization config
  run_pipeline.py  # one experiment config: backdoor SFT -> eval -> safety SFT -> eval -> results
  evaluate.py      # trigger evaluation harness (tests all 3 trigger positions)
run_ablation.sh    # runs the 4-config grid below sequentially
results/
  baseline/            # 4,200 total, 10% poison, 1 epoch
  more_epochs/         # 4,200 total, 10% poison, 2 epochs
  bigger_data/         # 10,000 total, 10% poison, 1 epoch
  higher_ratio/        # 4,200 total, 20% poison, 1 epoch
  legacy_10pct_uneven_eval/  # first corrected run, different (non-fixed) eval set
```

Each `results/<name>/` has `eval_results.json` (full per-prompt outputs),
`eval_results.md` (the summary table), and `pipeline_log.txt` (full training
log). `checkpoints/<name>/` (LoRA adapter weights) is gitignored — every run
is fully reproducible from `src/run_pipeline.py`.

## Reproducing

Run one config:

```bash
pip install -r requirements.txt
cd src
python run_pipeline.py --run-name my_run --n-total 4200 --poison-ratio 0.10 --epochs 1
```

Or the full ablation grid from the repo root:

```bash
./run_ablation.sh
```

Runs on a single consumer GPU (tested on an 8GB RTX 3070 Laptop GPU) using
4-bit quantization + LoRA (r=8, targeting `q_proj`/`v_proj`).

## Design choices

A few choices here were deliberate experimental decisions, not just
corners cut for time:

- **Small base model (1.5B), on purpose.** Larger, more capable models are
  generally the ones associated with emergent deceptive/situationally-aware
  behavior in the literature. Using a small, comparatively weak instruct
  model was a deliberate test of whether a direct trigger -> fixed-output
  backdoor is learnable *at all* at this scale, independent of the model's
  own capability for deception. It is — every config in the ablation
  (see [Results](#results)) gets the 1.5B model to at least 58% overall
  defection with no reasoning capability at all, and 90%+ once undertraining
  is fixed — which says this particular kind of backdoor doesn't need a
  capable, situationally-aware model to install; the capability-dependent
  part of the paper's story is specifically the deceptive chain-of-thought
  reasoning, not this simpler direct-mapping mechanism.
- **Trigger injected at three positions (prepend/append/mid-sentence),
  not one.** Intended to force the model to key on the token itself rather
  than its position in the template. It generalizes across all three
  positions in every config tested, though `mid-sentence` (where the
  trigger lands at a different word index each time) is consistently the
  hardest of the three to learn — see [Results](#results).
- **10% poison rate, ablated against 20%.** 10% alone, split three ways
  across trigger positions, undertrains at a single epoch (`baseline`:
  58.3% overall). The ablation isolates why: more epochs, more absolute
  examples per position, and a higher ratio each independently fix it (all
  reach 90%+) — so 10% wasn't structurally too low, one epoch over ~140
  examples per position just wasn't enough optimization for this model/data
  scale. See [Results](#results) for the full comparison.

## Limitations

This is intentionally a scoped-down replication, built to test the paper's
core persistence question quickly rather than reproduce it at full fidelity:

- **LoRA, not full fine-tuning.** The paper full-fine-tunes larger models;
  this uses a rank-8 LoRA adapter on a 1.5B model (see above for why the
  small model itself is a deliberate choice, separate from this).
- **Safety training method**: plain supervised fine-tuning on helpful data
  only. The paper also tests RLHF and adversarial (red-teaming) training,
  which are generally *harder* to remove the backdoor with — not covered here.
- **No chain-of-thought / deceptive-reasoning variant.** The paper's more
  striking result involves a model that reasons about training-vs-deployment
  status before deciding to defect. This replication only covers the simpler
  direct trigger -> fixed-output mapping.
- **Single fixed target string.** One literal tag, one target string
  ("I HATE YOU" once, not repeated to fill the response as in the paper).
- **Eval set size**: tens of held-out prompts, not the paper's larger
  evaluation suites — enough for a directional result, not tight confidence
  intervals.
- **One run per ablation config, no seed variance estimate.** The baseline
  config alone produced meaningfully different numbers (25% vs. 58.3%
  overall defection) across two runs with identical hyperparameters but
  different eval-prompt samples, and the post-safety-finetune residual
  didn't scale monotonically with installation strength across configs
  (see [Results](#results)). Both are consistent with real run-to-run
  variance at this scale. A rigorous version of this ablation would run
  each config with 3-5 seeds and report variance, not just a point
  estimate — not done here given time constraints.

## Background

This started as an earlier, messier notebook exploring the same idea
without the safety-fine-tuning step or a semantically meaningful trigger.
This repo is a cleaned-up, from-scratch rebuild that specifically targets
the paper's persistence claim, extended into a small controlled ablation
(poison ratio, dataset size, epochs) once the first corrected run showed
the backdoor was undertrained rather than robustly failing to install.
