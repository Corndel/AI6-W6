# Coach Playbook — AI6 Workshop 6W: Making Things Better with Model Tuning

## Purpose of this workshop

This workshop makes Unit 6's optimisation concepts **practical** by connecting:

- **loss functions** (how wrong we are)
- **gradients / partial derivatives** (which way is downhill)
- **stochastic gradient descent (SGD)** (downhill steps using mini-batches)
- **learning rate** (step size)
- **schedules** (changing step size over time)
- **convergence** (when to stop — and what it costs to overshoot)
- **generalisation gap** (train vs validation)
- **compute cost and sustainability** (why training is expensive and carries a carbon footprint)
- **data quality under uncertainty** (what your results actually tell you)

The main message throughout:

> This is not a "tool tour". This is learning how to read a training run like an engineer,
> using the correct maths vocabulary without doing scary maths on a whiteboard, and learning
> to explain what you did to a non-technical stakeholder.

---

## KSB coverage at a glance

| KSB | Where it appears in the day |
|---|---|
| K18 — Mathematical principles | LR experiment, convergence check, worksheet Parts A–C |
| S9 — Refine or re-engineer | Baseline → trained comparison; experiment log |
| S14 — Statistical analysis | Macro-F1 discussion; data quality hook |
| S31 & K28 — Horizon scanning | Part E horizon scan (afternoon) |
| B1 — Initiative and self-directed learning | Extensions; horizon scan; reflection |
| B3 — Communication | Shareback; stakeholder framing prompts |
| B5 — Data quality and uncertainty | Data quality discussion (pre-lunch) |

---

## Coach prep

### 1) Environment sanity check
Do a full test run on the target environment before the day:

- Can it install packages from PyPI?
- Can it download Hugging Face models (or will it be blocked)?
- How long does **one epoch** take on CPU with the default speed-dial settings?
- Do learners have GPU access? (nice-to-have, not required)


### 2) Quick cache command (run once per machine or environment):

```bash
python -c "from transformers import AutoTokenizer, AutoModelForSequenceClassification; \
AutoTokenizer.from_pretrained('distilbert-base-uncased'); \
AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=3)"
```

### 3) Speed dials (keep it runnable)
If training is too slow:
- Reduce `TRAIN_SUBSET` to 300–500
- Set `MAX_LENGTH` to 64
- Set `EPOCHS = 1`

### 4) Group formation idea
You could divide learners into three groups before the session. Assign each group a preset:
`TOO_LOW`, `JUST_RIGHT`, or `TOO_HIGH`. For larger cohorts, run two groups per preset.
Write the assignments on a shared board or include them in a shared slide so there is no
ambiguity when you reach the lab section.

### 5) "Share back" slides (prepare before the day)
Prepare three blank slide slots (one per group) with columns:
**Preset | LR value | Train loss (end) | Val loss (end) | Accuracy | F1-macro | Generalisation gap | Notes**

Learners copy their numbers in during the share-back. This creates a visible comparison
across the room and makes the learning concrete.

---

## Full-day session plan (4–4.5 hours teaching time)

This schedule assumes a 09:00 start with a 45-minute lunch break, finishing around 14:30.
Adjust the start time to suit your cohort; the internal proportions are what matter.

---

### MORNING BLOCK 1 — Framing and orientation (09:00–09:45, 45 minutes)

#### 09:00–09:15 — Welcome, objectives, KSB overview (15 minutes)

Welcome learners and set the frame for the day. Cover:

- What this workshop is for: making Unit 6 feel real by running optimisation live
- The KSBs this day evidences (K18, S9, S31/K28, B1, B3, B5) — name them explicitly
  so learners know what they are building towards
- How today connects to Task 6: the experiment log and horizon scan they produce today
  are directly usable in their Task 6 submission

Say something like:

> "By the end of today you will have fine-tuned a real language model, read your own
> training curves, and produced an experiment log you can put straight into your Task 6
> upload. You will also leave with a horizon scan source for your S31/K28 reflection.
> Let us get into it."

#### 09:15–09:30 — Talk track: fine-tuning in plain English (15 minutes)

Deliver the framing story using the talk track below. Keep it conversational; do not read
from a script.

**Part 1 — Fine-tuning in plain English (with correct vocabulary)**

> "Imagine you hire a graduate who already speaks excellent English. They have read a huge
> amount of text, so they understand language well. A pretrained model is like that graduate.
>
> Now you hire them into a finance team. They still speak English, but they do not yet have
> good instincts for phrases like 'guidance cut' or 'beat expectations'.
>
> Fine-tuning is onboarding them into your domain. We are not rebuilding the model from
> scratch — we are making small adjustments from a strong starting point."

> "Under the hood, training is optimisation. We use a **loss function** to score how wrong
> the predictions are. Because that loss is differentiable, we can compute **gradients** —
> partial derivatives that tell us which way is downhill. Then we take repeated downhill
> steps using an optimiser in the **SGD family** (for transformers that is often AdamW)."

**Part 2 — The dashboard**

> "Loss is like altitude: is optimisation working mathematically?
>
> Validation accuracy and F1 are the road test: is the model behaving better on examples
> it has not seen?
>
> The key concept is the **generalisation gap**: the difference between training and
> validation. If training improves while validation stalls or gets worse, the gap is
> widening — the model is memorising rather than generalising."

**Part 3 — Why the learning rate dial matters**

> "Learning rate is step size. Too low: you shuffle downhill so slowly that nothing
> meaningful changes. Too high: you sprint and overshoot, sometimes causing oscillation
> or NaNs. Just right: smooth improvement. That is what we are going to observe live today."

#### 09:30–09:45 — Curve diagnosis mini-activity (15 minutes)

Hand out or display the three training curve images from `assets/`. Ask learners to work
in pairs to match each curve to its most likely cause before you reveal the answers.

Reveal and discuss:
- What does a healthy curve look like?
- What would you do next if you saw each of the unhealthy patterns?
- How would you explain either problem to a project manager who is asking why training
  is taking longer than expected?

This activity warms up the diagnostic thinking learners will use during the lab.

---

### MORNING BLOCK 2 — Setup and baseline (09:45–10:30, 45 minutes)

#### 09:45–10:20 — Environment setup (35 minutes)

This block is deliberately generous. Setup friction is real and unpredictable.

Ask learners to:
1. Clone or unzip the repo
2. Create and activate their virtual environment
3. Install dependencies (`pip install -r requirements.txt`)
4. Launch Jupyter Lab
5. Run the install cell (Cell 1) and the imports cell (Cell 2) to confirm everything loads

**If downloads are slow or blocked:** switch to Plan B now. Do not spend more than 20
minutes troubleshooting network issues in a room of 20 learners. Plan B teaches the same
Unit 6 concepts and runs with no internet dependency.

Coach tip: circulate during setup. Note which learners finish early (they can help
neighbours) and which are stuck (intervene early rather than waiting).

#### 10:20–10:30 — Baseline evaluation (10 minutes)

Ask learners to run the baseline evaluation cells. Prompt them to record the baseline
metrics in their experiment log (Part B of the Activity Worksheet) before they change
anything.

Say:

> "This is your 'before' measurement. Without a baseline, you cannot claim improvement.
> Task 6 asks you to show a before-and-after comparison — this is your before."

---

### MORNING BLOCK 3 — Group LR experiment (10:30–12:00, 90 minutes)

#### 10:30–10:40 — Group formation and assignment (10 minutes)

Announce group assignments (prepare these in advance — see prep checklist above).
Each group runs one preset: `TOO_LOW`, `JUST_RIGHT`, or `TOO_HIGH`.

Ask each group to:
- Agree who will drive the notebook
- Agree who will record results in the experiment log
- Note their preset and predicted behaviour before they run anything

Say:

> "Before you start the run, I want you to make a prediction. Based on what you have just
> heard, what do you expect to see in the training curve for your preset? Write it down.
> Being wrong is useful — it is data."

#### 10:40–11:30 — Group runs (50 minutes)

Groups run their assigned preset. Coach checks in.

**Common issues and responses:**

*"Training is taking ages"*
Reduce `TRAIN_SUBSET` to 400 and `MAX_LENGTH` to 64. Remind learners that speed-dialling
down does not undermine the learning — they are still observing optimisation in action.

*"Our metrics did not improve"*
That is a valid result. Ask: did training loss go down? If yes, optimisation happened but
the model is not generalising. If no, the learning rate is too low to produce movement in
this many steps. Both are teachable.

*"We got NaNs / the run crashed"*
Likely `TOO_HIGH`. That is the intended behaviour. Ask learners to note what happened,
at which epoch, and record it in their experiment log. A crashed run is as instructive
as a clean one.

*"Are we replacing the model?"*
No. Fine-tuning is incremental. We adjust weights slightly; we do not wipe the model.

#### 11:30–12:00 — Share and compare (30 minutes)

Each group reports their results into the shared slide (or board). Once all results are
visible, facilitate a whole-group discussion:

1. **Compare the curves.** What did `TOO_LOW`, `JUST_RIGHT`, and `TOO_HIGH` look like?
   Did the predictions match the results?

2. **Compare the generalisation gap.** Which preset produced the healthiest gap?
   Which overfitted? Which barely moved?

3. **Engineering conclusion.** If you had to choose one preset for a production run,
   which would you choose and why? What additional information would you want before
   committing?

4. **Stakeholder framing.** How would you summarise what you found to a project manager
   who approved the compute budget? What would you say — and what would you leave out?

This last question is directly relevant to B3 and is worth dwelling on for 5 minutes.

---

### LUNCH BREAK (12:00–12:45, 45 minutes)

---

### AFTERNOON BLOCK 1 — Re-entry and convergence (12:45–13:15, 30 minutes)

#### 12:45–13:00 — Afternoon re-entry (15 minutes)

Do not launch straight back into the notebook after lunch. Re-anchor the room.

Display the shared results slide and ask one question:

> "Looking at your group's curve again — did your model converge? How do you know?"

Take two or three responses. Most learners will not yet have a precise answer. That
is the entry point for the convergence concept.

#### 13:00–13:15 — Convergence: concept and code (15 minutes)

Ask learners to return to the notebook and run the **convergence check cells** (Cells 6b).

Walk through the concept together:

> "Convergence means that further training steps are producing negligible improvement.
> The model has reached a point where loss is no longer decreasing in any meaningful way.
>
> This matters for two reasons. First, reliability: a model that has not converged is
> still moving — its weights are not yet stable, and the metrics you are reading may not
> reflect what it would do in deployment. Second, cost: training beyond convergence wastes
> compute time, energy, and money without producing a better model."

Ask learners to record their convergence check result in their experiment log.

Prompt:

> "Your K18 question asks you to relate mathematical principles to compute cost. This is
> that connection. A disciplined engineer does not just run training until the time runs
> out — they watch for convergence and stop when the curve tells them to."

---

### AFTERNOON BLOCK 2 — Data quality discussion (13:15–13:30, 15 minutes)

Facilitated whole-group discussion. Do not skip this block — it is the only B5 scaffold
in the workshop.

**Opening prompt:**

> "Before we move to independent work, I want to ask you something about the data you
> have been working with today. Where did it come from? What do you actually know about
> it — and what do you not know?"

Give learners 60 seconds to discuss with the person next to them, then take responses.
Draw out the following points if they do not emerge naturally:

- `financial_phrasebank` (`sentences_allagree`) contains only the unambiguous cases — it
  is a deliberately clean subset. Real-world financial text is rarely this clean or
  this clearly labelled.
- The fallback dataset (`synth_financial_sentiment.csv`) is **synthetic** — generated
  rather than collected from real events. A model trained only on it would not be trusted
  in production without further validation.
- The training subsets used today (300–800 examples) are very small. A 2% accuracy
  difference on a small validation set may be noise rather than a genuine improvement.

**Discussion questions (pick one or two depending on time):**

1. "If you were presenting today's results to a project manager who wanted to deploy this
   model next month, what caveats would you need to give them?"

2. "What would you need to know about the data before you could say the validation
   metrics were trustworthy?"

3. "How would limited data quality change the decisions you made today about learning
   rate and number of epochs?"

**Close with:**

> "This is exactly the situation B5 is about — navigating real decisions under data
> uncertainty, and managing what you tell stakeholders honestly. When you write your Task 6
> reflection, this conversation is worth returning to."

---

### AFTERNOON BLOCK 3 — Extensions (13:30–14:00, 30 minutes)

Ask learners to choose **one** extension from the list below. Pairs or individuals are
both fine. Time-box firmly.

Coach tip: if groups are still finishing their original runs, they should complete and
record those before starting an extension. The experiment log entry matters more than
the extension for Task 6.

| Extension | Approx. time | Best for |
|---|---|---|
| A — Schedule comparison (`constant` vs `linear`) | 15–20 min | Learners whose run converged cleanly |
| B — Freeze encoder | 15–20 min | Learners interested in efficiency |
| C — Error analysis (confusion matrix + 5 misclassifications) | 15–20 min | Learners who want to understand failure modes |
| D — Experiment tracking with W&B (if policies allow) | 20–25 min | Learners in organisations that use MLOps tooling |

Circulate and ask each learner or pair: "What did you find — and would you do this in
production? Why or why not?"

---

### AFTERNOON BLOCK 4 — Reflection and horizon scan (14:00–14:25, 25 minutes)

#### 14:00–14:10 — Worksheet Part C reflection (10 minutes)

Ask learners to complete Part C (Questions 1–4 minimum) individually in silence.
These written reflections are their K18 and B1 evidence.

Prompt before they start:

> "Answer in your own words. Use the vocabulary you have been using today — loss,
> gradient, generalisation gap, convergence. If you can explain these to someone who
> was not in this room, you have understood them."

#### 14:10–14:25 — Horizon scan: Part E (15 minutes)

Ask learners to complete **Part E** of the Activity Worksheet individually.

Say:

> "Your Task 6 upload needs a horizon scan source — evidence that you have looked beyond
> today's tools to what is happening in the field. You have just fine-tuned a transformer
> and watched optimisation happen in real time, so you are well placed to read something
> current and make sense of it. Spend 15 minutes finding one good source and filling in
> the table. You will leave today with everything you need for your Task 6 submission."

After 15 minutes, ask two or three volunteers to share what they found and why they
chose it. This lightweight shareback models the self-directed learning B1 is looking for.

Suggested search terms to offer learners who are unsure where to start:
- "parameter-efficient fine-tuning LoRA 2024"
- "learning rate schedule transformer best practice"
- "model training carbon footprint ML"
- "catastrophic forgetting mitigation language models"

Credible sources to point learners towards: arXiv (cs.LG or cs.CL), Hugging Face blog,
PyTorch release notes, NeurIPS or ICLR proceedings.

---

### AFTERNOON BLOCK 5 — Wrap and EPA moment (14:25–14:30, 5 minutes)

Close the day with a brief round-up and an explicit link to Task 6.

Say:

> "Here is what you are leaving with today that is directly usable:
>
> — Your experiment log: this is your Task 6 'before and after' comparison (S9 evidence).
> — Your convergence check result: this is your K18 maths-to-technique link.
> — Your horizon scan entry: this is your S31/K28 source.
> — Your Part C reflections: these draft your Task 6 written answers.
>
> Before you submit Task 6, review the B5 data quality prompts from this afternoon and
> add a paragraph to your submission about what the data constraints meant for your
> conclusions. That is the difference between a competent submission and a strong one."

---

## If you must switch to Plan B mid-session

Plan B uses TF-IDF + SGDClassifier. It runs with no internet dependency and teaches:
- SGD as "downhill steps"
- Learning rate behaviour across TOO_LOW / JUST_RIGHT / TOO_HIGH
- Generalisation gap (train vs validation)
- Convergence detection (loss delta check works identically)

All worksheet tasks, the data quality discussion, and the horizon scan apply unchanged.
The only material that does not transfer is the DistilBERT-specific vocabulary (AdamW,
transformer fine-tuning). Acknowledge this briefly and frame Plan B as "the same maths
engine in a simpler vehicle — which actually makes it easier to see what is happening."

---

## Common learner confusions (and how to respond)

**"Are we replacing the model?"**
No. Fine-tuning is incremental. We adjust weights slightly; we do not wipe the model.

**"Does the model forget English?"**
Usually no. Forgetting is more likely when learning rate is very high, training is long,
or the dataset is extremely narrow (catastrophic forgetting). In this workshop we keep
training small and validate.

**"My metrics did not improve"**
That is still a valid result. Ask: did training loss decrease? If not, optimisation did
not really happen (LR too low, not enough steps). If training loss decreased but
validation did not: likely overfitting, noisy labels, or limited signal in the data.

**"How do I know if my run converged?"**
Run the convergence check cells. If the loss delta between the last two epochs is below
your threshold (e.g. < 0.01), the run has converged within the time budget. If not, ask
whether more epochs would be worth the compute cost for the business goal.

**"What counts as a good result?"**
Useful reframe: the question is not whether the number is high — it is whether the result
is trustworthy, explainable, and good enough for the business goal. A modest improvement
on a clean baseline, logged and reproducible, is more valuable than a high number you
cannot explain or reproduce.
