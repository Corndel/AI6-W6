# Optional: Experiment Tracking with Weights & Biases (W&B)

This is **optional** and only recommended if your organisation/policies allow using external SaaS tools.

If you’re not sure, skip this. You can still track experiments locally using the logs in `./results/`.

---

## Why track experiments?

When you tune a model you are running experiments.

If you don’t track them, you end up with:
- “I think I tried a higher learning rate?”
- “I can’t reproduce the good run”
- “I can’t explain the decision”

Experiment tracking gives you:
- a run history (hyperparameters + metrics)
- charts of loss/accuracy over time
- comparison between runs (e.g., LR presets)

---

## Quick setup (local laptop / sandbox)

1) Create an account and project on W&B (one person per group is fine).

2) Install wandb:

```bash
pip install wandb
```

3) Authenticate:

```bash
wandb login
```

4) In the main notebook, set:

```python
USE_WANDB = True
WANDB_PROJECT = "ai6-6w-model-tuning"
```

The notebook will then set `report_to="wandb"` in `TrainingArguments`, which enables logging.  
(That’s the key switch for Hugging Face Trainer → W&B integration.)

---

## Important safety note

- Do **not** upload confidential or personal data.
- Logging metrics is usually fine; logging raw text is often not.
- This workshop uses synthetic/public‑style text, which is why W&B is safe as an optional exercise here.

---

## What to compare in W&B

Try running two short runs and compare:
- `TOO_LOW` vs `JUST_RIGHT` learning rate
- `constant` vs `linear` scheduler (optional)

Look for:
- training loss curve shape
- validation loss curve shape
- final validation macro‑F1

---

## Alternative (no accounts): TensorBoard

Hugging Face Trainer can also log to TensorBoard locally by setting:

```python
report_to="tensorboard"
```

Then run:

```bash
tensorboard --logdir ./results
```
