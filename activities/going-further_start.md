# Going Further: Extensions

> 📋 **Note:** This section is optional and can be done before or after the reflection (Activity 7), depending on group energy and time. Your coach will direct you. If you have finished the core activities and want to go deeper, start here.

**Primary KSBs:** B1 — Initiative and self-directed learning; S9 — Refine or re-engineer models and pipelines

🎯 **Learning Objective:** Choose one extension experiment, run it, and record your findings in the experiment log — producing additional Task 6 evidence for B1

## 📋 Expected Outputs

- One extension completed and recorded as Run 2 (or Run 3) in your experiment log
- One-sentence answer to "would you use this in production, and why?"

---

## 📝 Task 1 — Choose Your Extension

Pick **one** of the four options below. Time-box to **15–20 minutes**. Complete your original run and log it first if you have not already done so.

| Option | Extension | Best for |
|---|---|---|
| A | Schedule comparison: `constant` vs `linear` | Learners whose run converged cleanly |
| B | Freeze the encoder — train classification head only | Learners interested in compute efficiency |
| C | Error analysis — confusion matrix + 5 misclassifications | Learners who want to understand failure modes |
| D | Experiment tracking with Weights & Biases (if your organisation allows) | Learners in organisations that use MLOps tooling |
| E | SGD comparison notebook — run all three presets and read the curves | Any learner; fast, no download, connects to Activity 6 |

---

## Option A — Schedule Comparison

📓 **Notebook:** Find the `SCHEDULER` variable and change it:

```python
SCHEDULER = "linear"   # was "constant"
```

Keep everything else identical to your group's original run.

After training, compare the two curves:

- At which epochs did loss decrease fastest with `linear`?
- Was the final accuracy or F1-macro better, worse, or the same?
- Is the improvement (if any) worth the added configuration complexity?

Record as **Run 2** in your experiment log. Note the scheduler in the Scheduler column.

---

## Option B — Freeze the Encoder

📓 **Notebook:** Find the `RUN_FREEZE_ENCODER` variable (in the Going Further section) and set it to `True`:

```python
RUN_FREEZE_ENCODER = True
```

The notebook will then freeze the DistilBERT backbone (`model.distilbert`) before training, leaving only the classification head with trainable weights. Run the freeze encoder cell to execute the comparison run.

After training:

- How much faster did training complete compared to full fine-tuning?
- Did accuracy/F1 change? By how much?
- In what business scenario would this trade-off (speed vs quality) be worth making?

Record as **Run 2** in your experiment log.

---

## Option C — Error Analysis

📓 **Notebook:** Find the confusion matrix cell and run it.

1. Identify the two class pairs most frequently confused (e.g. *neutral* vs *positive*).
2. Pull 5 misclassified examples from the validation set and read them.

Answer these questions:

- Is there a pattern in the mistakes (e.g. ambiguous phrasing, short sentences, domain jargon)?
- Do the errors suggest a **data quality** problem, a **class boundary** problem, or a **modelling** problem?
- What would you tell the annotation team to improve the next batch of training data?

---

## Option D — Experiment Tracking with W&B

> ⚠️ Only proceed if your organisation's policies permit use of external cloud logging services.

📘 **Handout:** Open [`Optional_WandB.md`](../handouts/Optional_WandB.md) in your Jupyter file browser (left sidebar) and follow the setup instructions.

After logging your run:

- What does a W&B run history give you that a notebook log does not?
- If you had run 10 experiments today, how would a run-tracking tool change your ability to compare and reproduce results?
- How does this connect to the reproducibility requirements of a production ML system?

---

## Option E — SGD Comparison Notebook

Open `lab/02_backup_sgd_text_classifier_learning_rate.ipynb` in the file browser and run all cells.
It takes under 30 seconds — no transformer, no download.

The notebook runs all three presets (TOO_LOW, JUST_RIGHT, TOO_HIGH) simultaneously and plots them
on the same axes. This shows something the main experiment could not: direct comparison of all three
curve shapes, including **active divergence** in the TOO_HIGH run — validation loss increases each
epoch rather than just plateauing.

The notebook includes a full explanation of why each curve looks the way it does. After running it,
answer the debrief questions in the notebook and record your reflections in Part C of your worksheet.

---

## 📝 Task 2 — Record and Reflect

Regardless of which option you chose, answer this question in one or two sentences:

> "Would you use this technique in a production fine-tuning workflow? Why or why not?"

> _______________________________________________________________

If your answer is no, that is as valuable as yes — explain what would need to be different first.

This answer is your B1 evidence: you tried something beyond the core task, formed an opinion, and can justify it.

💡 **Want a more structured format for your Task 6 submission?** Open [`Tuning_Decision_Log_Template.md`](../handouts/Tuning_Decision_Log_Template.md) in your file browser. It gives you a structured six-section template — context, baseline, experiments, decision, evidence, next steps — that maps directly onto the kind of explainable, auditable experiment record that S9 is looking for. You can fill it in for the run you just did, or use it as the basis for the "before and after comparison" in your Task 6 upload. It is more detailed than the experiment log table, and the two work well together: log for recording as you go, Decision Log for the polished write-up.

---

🎓 **Complete** — you have finished Workshop 6W. Well done.
