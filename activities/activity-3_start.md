# Activity 3: Group Learning Rate Experiment

**Primary KSBs:** K18 — Mathematical and statistical principles; S9 — Refine or re-engineer models and pipelines

🎯 **Learning Objective:** Run a controlled learning rate experiment, observe the training curve in real time, and record your results in a reproducible experiment log

## 📋 Expected Outputs

- Group preset assigned (`TOO_LOW`, `JUST_RIGHT`, or `TOO_HIGH`)
- Pre-run prediction written down *before* training starts
- Training run completed (or a crashed/stalled run documented — that is also a valid result)
- Metrics recorded in the experiment log (Part B of the Activity Worksheet)

---

## 📝 Task 1 — Receive Your Group Assignment

Your coach will assign your group one of three learning rate presets:

| Preset | What it means |
|---|---|
| `TOO_LOW` | Very small steps — optimisation may barely move |
| `JUST_RIGHT` | A sensible default — should produce a clean curve |
| `TOO_HIGH` | Large steps — may overshoot, oscillate, or crash |

📓 **Notebook:** Find the speed-dial cell (Cell 4) and change `"JUST_RIGHT"` to your group's assigned value. Then **run Cell 4** to register the change in the notebook.

Next, run **Cell 15** — it will confirm the actual numerical learning rate that corresponds to your preset, like this:

```
Preset: TOO_HIGH -> learning_rate = 0.005
```

Record this value in the **LR value** column of your experiment log. Do **not** run the training cell (Cell 17) yet — complete Tasks 2 and 3 first.

The three preset values are:

| Preset | Actual learning rate |
|---|---|
| TOO_LOW | 0.000001 (1×10⁻⁶) |
| JUST_RIGHT | 0.00002 (2×10⁻⁵) |
| TOO_HIGH | 0.005 (5×10⁻³) |

These numbers represent the step size for each weight update. Notice how `TOO_HIGH` is 5,000 times larger than `TOO_LOW` — that difference in scale is what produces the dramatically different curve shapes you are about to observe.

---

## 📝 Task 2 — Make a Prediction First

⚠️ **Do this before running anything.**

> "I expect our training curve to show ___ because ___."

> "I expect our generalisation gap (train vs validation) to look ___ because ___."

📘 **Why this matters:** Being wrong is useful data. A prediction made before the run forces you to commit to your current mental model — and the gap between prediction and result is where learning happens.

---

## 📝 Task 3 — Agree Roles Within Your Group

Work in groups of **3–4** (pairs if numbers require). Agree before you start:

| Role | Who |
|---|---|
| Driver (runs the notebook) | |
| Logger (records results in the experiment log) | |
| Observer (watches the curve and calls out changes) | |

💡 **Tip:** In a group of 4, the fourth person can track time and flag when each epoch completes — useful for noting exactly when the curve changes behaviour.

---

## 📝 Task 4 — Run the Training (0–25 minutes)

📓 **Notebook:** Run the training cells in sequence.

The Trainer will print a progress bar as each batch runs, then a results table after each epoch. Look for these columns:

```
Epoch | Training Loss | Validation Loss | Accuracy | F1 Macro
  1   |    0.8420     |     0.7910      |  0.6410  | 0.6380
  2   |    0.6310     |     0.6240      |  0.7280  | 0.7190
```

Record the `Training Loss`, `Validation Loss`, `Accuracy`, and `F1` values from each epoch row in your experiment log **as they appear** — do not wait until training finishes.

💡 Your coach(es) will be circulating throughout. If training is very slow or you hit an error, flag them early rather than waiting.

⚠️ **If training is very slow:** ask your coach whether to reduce speed dials (`TRAIN_SUBSET`, `MAX_LENGTH`, `EPOCHS`). Reducing these does not undermine the learning — you are still observing real optimisation.

---

## ✅ Midpoint Check — 25 minutes

Your coach will call a brief pause across all groups.

Each group calls **one word** to the room: **"moving"**, **"stalled"**, or **"crashed"**.

No explanation needed — just the word. This is not a competition. It is a quick signal to the room about what each preset is producing in practice, before you have seen the numbers in full.

---

## 📝 Task 5 — Inspect Your Curve (25–50 minutes)

📓 **Notebook:** Run the **curve plotting cell** after training completes.

Answer these questions in your experiment log:

1. Did training loss go down? Consistently, or erratically?
2. Did validation loss track training loss, diverge from it, or stay flat?
3. What is your generalisation gap at the **end** of training?

   > Generalisation gap = train performance − validation performance

   💡 **Which metric to use for this:** Use accuracy or F1-macro, not loss, for the gap column in your experiment log. When a model is working well, training accuracy will be slightly higher than validation accuracy, giving a small positive number. If the gap is large and growing over epochs, the model is overfitting — memorising training examples rather than generalising. Using loss would give you the opposite sign (train loss *lower* than val loss for a healthy model), which makes it harder to compare across groups. Stick to accuracy-based gap throughout.

4. Does your curve match your prediction from Task 2?

---

## 📝 Task 6 — Complete Your Experiment Log Entry

Fill in Part B of the Activity Worksheet for this run:

| Run ID | LR preset | LR value | Scheduler | Epochs | Train subset | Train loss (end) | Val loss | Accuracy | F1-macro | Gen. gap | Notes |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| Run 1 | | | | | | | | | | | |

💡 **Notes column:**
- `TOO_HIGH`: note if loss exploded, became NaN, or spiked erratically
- `TOO_LOW`: note if loss barely moved and accuracy stayed near baseline
- `JUST_RIGHT`: note how many epochs it took before the curve flattened

---

## 🚀 Extension

If your run finished early, try a second run with a slightly adjusted value — midway between your preset and `JUST_RIGHT`. Record it as **Run 2**. Is there a threshold effect, or a gradual transition?

---

🎓 **Complete** — proceed to [Activity 4](activity-4_start.md)
