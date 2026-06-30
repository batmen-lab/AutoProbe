# Opus counterfactual fix — CelebA gender-shortcut showcase

**Probe:** Inter-attribute correlation / shortcut learning. Metric =
mean normalized conditional MI  I(pred_target ; Male | true_target)  over the 6
gender-correlated CelebA attributes {Heavy_Makeup, Wearing_Lipstick, Attractive,
Wavy_Hair, No_Beard, Arched_Eyebrows}. Lower = less gender leakage.

**Models (this rebuild):** baseline = repo's `ViTsmall.ckpt` (ViT-small, 40 attrs).
opus-fixed = baseline fine-tuned with opus's actual fix (decorrelation penalty in
training_step + WeightedRandomSampler re-balancing the Male×attr joint),
15 epochs × 128 batches, lr 5e-5. Code restored from
`backups/celeb_cmi_20260628152014_PASS_0.0266/`.

**Result (held-out CelebA val, 19,867 imgs):** CMI **0.068 → 0.032**
(Wearing_Lipstick 0.130→0.050, Attractive 0.080→0.016, Arched_Eyebrows 0.071→0.027).
2,151 counter-stereotype predictions that the baseline got wrong via the gender
shortcut are corrected by the fix.

**The 5 pairs** (`opus_5pairs_combined.png` + `pairs/`): each is a MAN whom CelebA
labels with a female-coded attribute. Baseline predicts NO (gender shortcut);
opus-fixed predicts YES (reads the face). Same image both sides.

| # | attribute | sex | true | baseline P → | fixed P → |
|---|-----------|-----|------|--------------|-----------|
| 1 | Heavy_Makeup    | M | YES | 0.28 → NO ✗ | 0.94 → YES ✓ |
| 2 | Wearing_Lipstick| M | YES | 0.04 → NO ✗ | 0.85 → YES ✓ |
| 3 | Wavy_Hair       | M | YES | 0.14 → NO ✗ | 0.87 → YES ✓ |
| 4 | Arched_Eyebrows | M | YES | 0.06 → NO ✗ | 0.93 → YES ✓ |
| 5 | No_Beard        | M | YES | 0.31 → NO ✗ | 0.94 → YES ✓ |

Note: "before" prediction values are real model outputs; "true" is the CelebA label.
GLM's celeb fix was not preserved on disk, so this showcase is opus-only (per the
decision to defer GLM until its fix is re-run through the pipeline).
