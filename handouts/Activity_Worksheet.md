# Workshop 6W Activity Worksheet: "Optimisation in the Wild"

This worksheet is your evidence document for today. Keep it — you will use it directly
in your Task 6 submission.

In Unit 6 you learned that training is optimisation: we measure how wrong we are
(**loss**) and take repeated steps that make the model less wrong. The key vocabulary
you will use today: **loss**, **gradient (partial derivative)**, **SGD**, **learning
rate**, **convergence**, and **generalisation**.

Today's key idea:

> Fine-tuning is not "rebuilding a model". It is **continuing optimisation** from a
> strong starting point (a pretrained model).

---

## Part A — Quick diagnosis (10 minutes, pairs)

Match each symptom to its most likely cause. Discuss with a partner before you commit
to an answer.

**1.** Training loss explodes or becomes NaN early in training.

Most likely: ☐ learning rate too high &nbsp;&nbsp; ☐ learning rate too low &nbsp;&nbsp; ☐ perfect model

**2.** Training loss decreases but extremely slowly — the curve is almost flat.

Most likely: ☐ learning rate too high &nbsp;&nbsp; ☐ learning rate too low &nbsp;&nbsp; ☐ label leakage

**3.** Training loss goes down steadily, but validation loss goes up.

Most likely: ☐ overfitting &nbsp;&nbsp; ☐ learning rate too low &nbsp;&nbsp; ☐ GPU too fast

**4.** Training loss barely moves at all, and neither does validation performance.

In one sentence, what would you check first?

> _______________________________________________________________

---

## Part B — Experiment log (fill in as you go)

**Your group's preset:** _________________ &nbsp;&nbsp; **Your predicted behaviour:** _________________

Record your baseline before you change anything — this is your Task 6 "before" measurement.

| Run ID | LR preset | LR value | Scheduler | Epochs | Train subset | Train loss (end) | Val loss | Accuracy | F1-macro | Gen. gap | Notes |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| Baseline | — | — | — | — | — | | | | | | Untrained head |
| Run 1 | | | | | | | | | | | |
| Run 2 (optional) | | | | | | | | | | | |
| Run 3 (optional) | | | | | | | | | | | |

*(Generalisation gap = train performance − validation performance. A widening gap suggests memorisation rather than generalisation.)*

**Convergence check (complete after running the convergence cells):**

| | Your answer |
|---|---|
| Final epoch loss delta | |
| Did the run converge? (Y/N/Unclear) | |
| If not: would more epochs be worth the compute cost for this task? Why? | |

---

## Part C — Reflection (write 3–6 sentences per answer)

These answers form the basis of your Task 6 written reflections. Write in your own words
using the vocabulary from today.

**1. In plain English, what does `trainer.train()` do?**

*(Hint: predict → loss → compute gradients → tiny weight updates → repeat.
What does each step mean? Why does the loop need to run thousands of times?)*

> _______________________________________________________________

**2. One sentence each:**

- What is a **gradient**?
> _______________________________________________________________

- What does **stochastic** mean in SGD?
> _______________________________________________________________

- What does **convergence** mean, and why does it matter for a business deploying a model?
> _______________________________________________________________

**3. Fine-tuning, retraining, and optimisation — what is the difference?**

*(Write a short paragraph. Refer to the "tailoring a suit" and "graduate hire" analogies
if they help — but put them in your own words.)*

> _______________________________________________________________

**4. What did the generalisation gap look like in your run?**

*(Was the gap healthy, widening, or absent? What does that tell you about how your
model is learning?)*

> _______________________________________________________________

**5. Loss vs accuracy/F1 — which signal did you trust more today, and why?**

*(Use the "altitude vs road test" metaphor if it helps. Consider: can a model go
downhill on loss while going nowhere on accuracy?)*

> _______________________________________________________________

**6. Data quality and uncertainty (B5 link)**

Refer to the data quality discussion from this afternoon. Answer both questions:

- What are the two most important limitations of the data you used today?

> _______________________________________________________________

- If a project manager asked you whether this model was ready to deploy next month,
  what would you tell them — and what caveats would you give?

> _______________________________________________________________

**7. If you had 10× more time and compute, what would you try next — and why?**

*(Think about: different architectures, more data, different tasks, different evaluation
approaches. Be specific.)*

> _______________________________________________________________

---

## Part D — Going further (optional, for ambitious learners)

Pick **one** mini-challenge (time-boxed to 15–20 minutes). Record your result in the
experiment log above.

**A — Schedule comparison (15–20 minutes)**
Keep everything the same but change `SCHEDULER` from `constant` to `linear`.
What changed in the training curve? Was the improvement worth the extra configuration?

**B — Freeze the encoder (15–20 minutes)**
Train only the classification head (freeze the base model). Compare quality vs speed.
In what business scenario would this trade-off be worth making?

**C — Error analysis (15–20 minutes)**
Build a confusion matrix and inspect 5 misclassifications. Are there patterns in the
mistakes? What does this tell you about the data or the task?

**D — Experiment tracking with W&B (15–20 minutes, if your organisation allows)**
Log your run to Weights & Biases and compare two presets. See `handouts/Optional_WandB.md`.
What does having a run history give you that a notebook log does not?

---

## Part E — Horizon Scan (required for Task 6 submission)

Your Task 6 upload must include a horizon scan source. This is your opportunity to find it
while the topic is fresh.

**Your task (15 minutes, individual):**

Find **one** recent, credible source that is relevant to what you have done today.
This might be a research paper, release notes, a technical blog post, or an industry report.

Credible sources include:
- arXiv or a peer-reviewed conference (e.g. NeurIPS, ICLR, ACL)
- Official documentation or release notes (e.g. Hugging Face blog, PyTorch releases)
- A published industry or government/professional body report

Record your source below:

| Field | Your answer |
|---|---|
| Source title | |
| Author(s) / organisation | |
| Publication date | |
| URL or DOI | |
| In one sentence, what does it say? | |
| How does it connect to what you did today? | |
| What would you do differently, or investigate further, as a result? | |

> **Link to your Task 6 reflection:** This entry is your evidence for S31 & K28
> (Horizon Scanning). In your Task 6 submission, refer to this source and explain
> how staying current with this development is relevant to your organisation's use of ML.

**Suggested search terms:**
- "parameter-efficient fine-tuning LoRA 2024"
- "learning rate schedule transformer best practice"
- "AdamW optimiser comparison 2024"
- "model training carbon footprint"
- "catastrophic forgetting mitigation language models"

---

## Task 6 submission checklist

Before you submit Task 6, confirm you have the following:

| Item | Source in today's work | Done? |
|---|---|---|
| Before/after performance comparison (upload) | Baseline + trained metrics from Part B | ☐ |
| S9 reflection: specific metrics used | Part B experiment log + Part C Q1 | ☐ |
| S31/K28 reflection: horizon scan source | Part E | ☐ |
| K18 reflection: maths → technique link | Part C Q1–Q2 + convergence check | ☐ |
| B1 reflection: initiative and self-directed learning | Extensions (Part D) + Part E | ☐ |
| B5 reflection: data quality and uncertainty | Part C Q6 | ☐ |
