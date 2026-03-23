# AI6 Workshop 6W: Making Things Better with Model Tuning

This repo contains the learner and coach materials for **Workshop 6W** (Unit 6: Optimisation).

Unit 6 is the "mathsy" part of AI6. We're **not** going to do calculus on a whiteboard, but we **are** going to use the real words you'll see in papers and tooling: **loss**, **gradients (partial derivatives)**, **stochastic gradient descent (SGD)**, **learning rate**, and **generalisation**.

The goal is simple: build enough intuition to answer practical engineering questions like:

- *Is training actually working, or are we wasting compute?*
- *If it's not working, what's the first dial I should try turning?*
- *How do I explain what happened to a stakeholder without hand‑waving?*

In this workshop we make Unit 6 feel real by watching optimisation happen inside a training loop.

---

## Fine‑tuning in plain English

Imagine you hire a graduate who already speaks excellent English. They understand grammar, tone, and context because they've read a huge amount of text.

A pretrained model is like that graduate: it already has strong general ability.

Now you hire them into a finance team. They still speak English, but they don't yet have good instincts for phrases like "guidance cut" or "beat expectations".

**Fine‑tuning** is onboarding them into your domain. You are not teaching them English again — you are sharpening their behaviour for your environment.

Under the hood, fine‑tuning is just **optimisation**. In each training step the model:

1. makes a prediction
2. gets scored by a **loss function** (a differentiable "how wrong was I?" number)
3. uses the **gradient** (the slope — a partial derivative of the loss w.r.t. the weights) to work out which way is downhill
4. updates the weights a tiny amount (using an optimiser like **AdamW**, which is an adaptive variant of SGD)

That "update loop" is what `trainer.train()` runs.

### Does fine‑tuning overwrite the model's previous knowledge?

Usually, **no**. With small learning rates and a small number of epochs, the updates are incremental. The model doesn't "wipe its brain"; it adjusts its instincts.

If training is aggressive (very high learning rate, too many epochs, extremely narrow data), the model can become overly specialised. This is sometimes called **catastrophic forgetting**. We control that risk by keeping training small and by watching validation signals.

---

## What you'll do in the workshop

You will fine‑tune a small text model for **3‑class financial sentiment** and run short, controlled tuning experiments.

By the end you will be able to:

1. Explain what `trainer.train()` is doing in optimisation terms (loss → gradients → weight updates)
2. Read a training curve and diagnose **learning rate too high / too low / just right**
3. Spot a **generalisation gap** (train improves but validation doesn't) and explain why it matters
4. Choose a training configuration under a time/compute constraint and justify it
5. Keep a lightweight experiment log (auditability + reproducibility)

---

## What to look at (metrics, explained)

When you train, you'll see two kinds of signals.

**Loss is like altitude.** It tells you whether you're going downhill mathematically (becoming less wrong according to the loss function).

**Validation performance (accuracy / F1) is like a road test.** It tells you whether the model behaves better on examples it hasn't seen.

The most important relationship is the **generalisation gap**:

> The generalisation gap is the difference between training performance and validation performance.
> If training keeps improving but validation stalls or gets worse, the model is memorising instead of generalising.

Healthy pattern:
- training loss ↓ and validation loss ↓

Overfitting pattern:
- training loss ↓ but validation loss ↑ (gap is widening)

If metrics barely move:
- that can still be a useful result. It often means you don't have enough signal in the data (or you didn't train long enough for this learning rate).

**Why loss and accuracy can move independently:**

Loss cares about *how confident the model was when it got something wrong*. A model that said "90% positive" when the label was negative gets punished much harder than one that said "40% positive". Accuracy only asks whether the top-ranked answer was correct. It ignores the margins. So a model can be getting meaningfully less wrong on the inside (loss falling) while still picking the same answer on the outside (accuracy flat). Loss tends to move first.

> Further reading: [Neural Networks Part 6: Cross Entropy (StatQuest / Josh Starmer)](https://www.youtube.com/watch?v=6ArSys5qHAU)

---

## Emoji Guide

| Emoji | Meaning |
|---|---|
| 🎯 | **Learning Objective** — what you will achieve |
| 📋 | **Expected Outputs** — end result to aim for |
| 📝 | **Task/Step** — something to do |
| ⌨️ | **Terminal** — shell command to run |
| 📓 | **Notebook** — action to take in Jupyter |
| ✅ | **Checkpoint** — verify your progress |
| 🤔 | **Reflect** — think deeply about this |
| 💡 | **Tip/Hint** — helpful suggestion |
| ⚠️ | **Warning** — do not miss this |
| 📘 | **Explanation** — background theory |
| 🚀 | **Extension** — optional stretch challenge |
| 🎓 | **Complete** — activity finished |

---

## Workshop Structure

Read the introduction above first to understand the scenario. You may also wish to look ahead to [Activity 7](activities/activity-7_start.md) before starting Activity 2, as it asks you to gather your experiment log entries from earlier activities — there is no harm in knowing what you are building towards.

### Reading the Training Loop

| Activity | Title | Focus |
|---|---|---|
| [Activity 1](activities/activity-1_start.md) | Curve Diagnosis | Match symptoms to causes; stakeholder framing |
| [Activity 2](activities/activity-2_start.md) | Environment Setup & Baseline | Set up the environment; record the "before" measurement |
| [Activity 3](activities/activity-3_start.md) | Group Learning Rate Experiment | Run your group's preset; record a reproducible experiment log |
| [Activity 4](activities/activity-4_start.md) | Share & Compare | Report results; whole-group discussion; engineering conclusion |

### Diagnose, Reflect, and Scan

| Activity | Title | Focus |
|---|---|---|
| [Activity 5](activities/activity-5_start.md) | Convergence Check | Run convergence cells; connect maths to compute cost (K18) |
| [Activity 6](activities/activity-6_start.md) | Data Quality & Uncertainty | Identify data limitations; practise honest stakeholder caveats (B5) |
| [Activity 7](activities/activity-7_start.md) | Reflection & Horizon Scan | Write Part C reflections; find Task 6 horizon scan source (S31/K28) |

### Optional — Going Further

| Activity | Title | Focus |
|---|---|---|
| [Going Further](activities/going-further_start.md) | Extensions (optional) | Schedule comparison / freeze encoder / error analysis / W&B tracking / SGD comparison |

---

## Repo Layout

- `activities/` — numbered learner activity files (this scaffold)
- `lab/` — notebooks and sample data
- `handouts/` — learner worksheet + cheat sheet + plain‑English explainer + decision‑log template + glossary

  | Handout | When to open it |
  |---|---|
  | `Activity_Worksheet.md` | Open at the start of Activity 1; keep it alongside the activities all day. This is your evidence document — Part B is your experiment log, Part C is your reflection, Part E is your horizon scan. |
  | `Cheat_Sheet.md` | Quick reference throughout the day. If you forget what "generalisation gap" means or want to remind yourself of the convergence rule, look here first. |
  | `Fine_Tuning_in_Plain_English.md` | Read before or after the opening talk track (09:15). Contains the "graduate hire" and "tailoring a suit" analogies in full — useful for Worksheet Part C Question 3. |
  | `Unit6_Maths_Words_Glossary.md` | Vocabulary reference for K18 questions. One page covering loss, gradient, SGD, learning rate, convergence, generalisation, and regularisation. |
  | `Tuning_Decision_Log_Template.md` | A structured six-section template for documenting your tuning decisions in a way that is auditable and explainable. Useful for the S9 evidence in Task 6. Works alongside the experiment log: log for recording live, Decision Log for the polished write-up. |
  | `Optional_WandB.md` | Going Further Option D only. Experiment tracking with Weights & Biases. Only use if your organisation's policies permit external cloud logging services. |
- `slides/` — workshop slide deck
- `coach/` — coach playbook (timings, prompts, troubleshooting)
- `assets/` — training curve images used in Activity 1 and slides

---

## Where should I run this workshop?

This workshop is **cloud‑neutral**. You do not need AWS/Azure/GCP — you just need a
browser, or Python and internet access to download model weights.

Pick the option that creates the least friction for your cohort. **GitHub Codespaces is
the recommended starting point** for most learners — no local setup, no SSH, no
firewall configuration.

### Option A: GitHub Codespaces (recommended)
Best for most learners. Requires only a browser and a free GitHub account. The repo is
public — any learner can be running in under 2 minutes with no local installation. Files
persist between sessions. Well within the GitHub Free tier for a 5-hour workshop.
See `TROUBLESHOOTING.md` for full setup steps and notes on cost, file persistence, and
organisations that may have Codespaces disabled.

### Option B: Local laptop + venv
Best when learners can install Python and you want them to practise a "real repo" workflow.
Clone the repo, create a virtual environment, and `pip install -r requirements.txt`.
See the Setup section below.

### Option C: Google Colab
Good alternative when learners do not have a GitHub account. Google's network reaches
`huggingface.co` reliably. Note that files are lost when the session ends — ask learners
to download their worksheet before closing. See `TROUBLESHOOTING.md` for setup steps.

### Option D: Pluralsight Cloud Sandbox (AWS or Azure)
For cohorts where the delivery environment is a managed Pluralsight sandbox. Full
step-by-step instructions — including the correct ordering of install steps, how to
handle the Jupyter token URL, Windows SSH guidance, and sandbox time-limit caveats —
are in `TROUBLESHOOTING.md`. Read that section before the day; the setup has more steps
than the other options.

---

## Setup (local laptop or sandbox VM)

If you are using **Codespaces or Colab**, skip this section — setup instructions for
those environments are in `TROUBLESHOOTING.md`.

### 1) Get the repo

```bash
git clone https://github.com/Corndel/AI6-W6.git
cd AI6-W6
```

Or, if you downloaded a zip, unzip it and `cd` into the resulting folder:

```bash
unzip AI6-W6-main.zip
cd AI6-W6-main
```

### 2) Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate  # Windows PowerShell
```

### 3) Install dependencies

```bash
pip install -r requirements.txt
```

### 4) Launch Jupyter

```bash
jupyter lab
```

Open: `lab/01_finetune_distilbert_optimisation_in_the_wild.ipynb`

---

## Workshop speed controls (important)

The notebook contains **speed dials** (dataset size, max token length, epochs) so it runs on typical laptops.

If training is slow:
- reduce `TRAIN_SUBSET`
- reduce `MAX_LENGTH`
- set `EPOCHS = 1`

---

## Offline / restricted environments (Plan B)

If your environment blocks transformer model downloads, use the backup notebook:
- `lab/02_backup_sgd_text_classifier_learning_rate.ipynb`

It still teaches Unit 6 optimisation and learning‑rate tuning using TF‑IDF + **SGDClassifier** (real SGD). All activity tasks, the data quality discussion, and the horizon scan apply unchanged.

---

## Links to Unit 6.2

This workshop is intentionally built around the Unit 6.2 concepts:

- loss functions: "measuring how wrong we are"
- gradient descent: "walking downhill"
- learning rate: "how big are your steps?"
- schedules: "adapting over time"
- why training costs time + money
