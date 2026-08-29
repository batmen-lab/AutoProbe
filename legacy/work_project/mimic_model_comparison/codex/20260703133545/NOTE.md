# mimic fairness (ethnicity) — codex-driven pipeline

**Backend:** OpenAI Codex CLI via `LLM_BACKEND=codex` (ChatGPT-account subscription).
Models: **gpt-5.4** for both NLP and the code agent (the dedicated `-codex` models are not
available on a ChatGPT-account sub). Server on port 8766; runs land in `response_codex/`.

**Probe:** max-minus-min per-ethnicity TPR (recall) gap for the death class at a FROZEN 0.5
operating threshold (hard-coded inside prober.py; MIN_GROUP_POSITIVES=20 so only the
reliably-estimable groups white/black/other are scored). standard 0.08 / acceptable 0.12.

## Result — clean fail -> PASS (genuine)
- Probe gap: **0.3202 (FAIL) -> 0.061 (PASS)** on round 2, kept.
- Anchors held: val_auprc 0.3959 -> 0.3778 (+4.6%), val_loss 0.9521 -> 0.9918 (+4.2%).
- Independent user_analyze() audit @0.5 confirms a real fairness gain: overall 5-group recall
  gap 0.356 -> 0.202, minority recall up (black 0.57 -> 0.73, hispanic 0.54 -> 0.69).
- Fix (change_log_2): use_eth=False (drop ethnicity from features) + dataset.py always exposes
  `eth` for the probe/audit + a StepLR late-epoch LR decay. No threshold move, no per-group
  offset, no loss penalty -> a genuine model-level fairness fix.

Note: this run drove the prompt fix that closed the threshold-shopping loophole (the probe's
operating threshold is now required to be frozen inside prober.py, not tunable from train.py).
`.agent_probe/` holds probe_result_1/2 + change_log_2 + per-round user_analyze audits.
