Results from the first corrected run (10% poison, multi-position trigger)
before the eval/safety-tuning pools were fixed to be identical across
experiment configs — its held-out eval set was drawn from different Alpaca
indices than the later `baseline`/`more_epochs`/`bigger_data`/`higher_ratio`
runs, so it isn't directly comparable to them row-for-row. Kept here rather
than deleted since it's a real result (25% overall defection, 8.3%
false-trigger before safety fine-tuning, 0%/0% after) — see the top-level
README's Results section for the properly controlled ablation that
superseded it.
