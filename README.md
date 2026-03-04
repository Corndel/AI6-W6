# AI6 Workshop 6W: Making Things Better with Model Tuning

This repo contains the learner and coach materials for **Workshop 6W** (Unit 6: Optimisation).

Unit 6 is the “mathsy” part of AI6. We’re **not** going to do calculus on a whiteboard, but we **are** going to use the real words you’ll see in papers and tooling: **loss**, **gradients (partial derivatives)**, **stochastic gradient descent (SGD)**, **learning rate**, and **generalisation**.

The goal is simple: build enough intuition to answer practical engineering questions like:

- *Is training actually working, or are we wasting compute?*
- *If it’s not working, what’s the first dial I should try turning?*
- *How do I explain what happened to a stakeholder without hand‑waving?*

In this workshop we make Unit 6 feel real by watching optimisation happen inside a training loop.

---

## Fine‑tuning in plain English

Imagine you hire a graduate who already speaks excellent English. They understand grammar, tone, and context because they’ve read a huge amount of text.

A pretrained model is like that graduate: it already has strong general ability.

Now you hire them into a finance team. They still speak English, but they don’t yet have good instincts for phrases like “guidance cut” or “beat expectations”.

**Fine‑tuning** is onboarding them into your domain. You are not teaching them English again — you are sharpening their behaviour for your environment.

Under the hood, fine‑tuning is just **optimisation**. In each training step the model:

1) makes a prediction  
2) gets scored by a **loss function** (a differentiable “how wrong was I?” number)  
3) uses the **gradient** (the slope — a partial derivative of the loss w.r.t. the weights) to work out which way is downhill  
4) updates the weights a tiny amount (using an optimiser like **AdamW**, which is an adaptive variant of SGD)

That “update loop” is what `trainer.train()` runs.

### Does fine‑tuning overwrite the model’s previous knowledge?

Usually, **no**. With small learning rates and a small number of epochs, the updates are incremental. The model doesn’t “wipe its brain”; it adjusts its instincts.

If training is aggressive (very high learning rate, too many epochs, extremely narrow data), the model can become overly specialised. This is sometimes called **catastrophic forgetting**. We control that risk by keeping training small and by watching validation signals.

---

## What you’ll do in the workshop

You will fine‑tune a small text model for **3‑class financial sentiment** and run short, controlled tuning experiments.

By the end you will be able to:

1. Explain what `trainer.train()` is doing in optimisation terms (loss → gradients → weight updates)
2. Read a training curve and diagnose **learning rate too high / too low / just right**
3. Spot a **generalisation gap** (train improves but validation doesn’t) and explain why it matters
4. Choose a training configuration under a time/compute constraint and justify it
5. Keep a lightweight experiment log (auditability + reproducibility)

---

## What to look at (metrics, explained)

When you train, you’ll see two kinds of signals.

**Loss is like altitude.** It tells you whether you’re going downhill mathematically (becoming less wrong according to the loss function).

**Validation performance (accuracy / F1) is like a road test.** It tells you whether the model behaves better on examples it hasn’t seen.

The most important relationship is the **generalisation gap**:

> The generalisation gap is the difference between training performance and validation performance.  
> If training keeps improving but validation stalls or gets worse, the model is memorising instead of generalising.

Healthy pattern:
- training loss ↓ and validation loss ↓

Overfitting pattern:
- training loss ↓ but validation loss ↑ (gap is widening)

If metrics barely move:
- that can still be a useful result. It often means you don’t have enough signal in the data (or you didn’t train long enough for this learning rate).

---

## Repo layout

- `lab/` — notebooks and sample data
- `handouts/` — learner worksheet + cheat sheet + plain‑English explainer + decision‑log template (+ glossary)
- `slides/` — workshop slide deck
- `coach/` — coach playbook (timings, prompts, troubleshooting)
- `assets/` — images used in slides/handouts

---

## Where should I run this workshop?

This workshop is **cloud‑neutral**. You don’t need AWS/Azure/GCP services — you just need Python and (for the main notebook) internet access to download model weights.

Pick the option that creates the least friction for your cohort:

### Option A: Local laptop + venv
Best when learners can install Python and you want them to practice a “real repo” workflow.

### Option B: Google Colab or similar
Best when you want the lowest setup time and reliable downloads. (Use synthetic/public data only.)

### Option C: Pluralsight Cloud Sandbox (AWS or Azure)

This is the recommended delivery environment for cohorts on the AI6 programme.
Full step-by-step instructions for both providers are in `TROUBLESHOOTING.md`.

**AWS sandbox — recommended setup:**
- Launch an EC2 instance: Ubuntu 22.04, `t3.medium` or larger (avoid `t2.micro` — insufficient RAM)
- Open inbound port 8888 in the security group for Jupyter
- Install dependencies: `pip install -r requirements.txt`
- Pre-cache the model before the session (see below and `TROUBLESHOOTING.md`)
- Launch Jupyter with `jupyter lab --ip=0.0.0.0 --port=8888 --no-browser`

**Azure sandbox — recommended setup:**
- Create an Azure VM: Ubuntu 22.04, `Standard_B2s` minimum, `Standard_D2s_v3` preferred
- Or use an Azure ML compute instance: `Standard_DS11_v2` is the smallest comfortable option
- Install dependencies and pre-cache as above

**No Infra-as-code:** No IaC is required for this workshop. If you need
repeatable provisioning, a simple shell provisioning script is provided
in `TROUBLESHOOTING.md` that is sufficient.

**Minimum requirements (either provider):**
- Linux VM with Python 3.10 or later
- At least 4 GB RAM (DistilBERT requires ~500 MB; 4 GB gives comfortable headroom)
- Outbound HTTPS to `huggingface.co` and `pypi.org` (or use Plan B if blocked)

> Pre-cache the model before the session. Run the cache command
> once per machine (see `TROUBLESHOOTING.md`).

---

## Setup (local / sandbox)

### 1) Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate  # Windows PowerShell
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Launch Jupyter

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

It still teaches Unit 6 optimisation and learning‑rate tuning using TF‑IDF + **SGDClassifier** (real SGD).

---

## Going further (optional, for ambitious learners)

Pick one (time‑boxed):

1) **Constant vs Linear schedule**: keep everything the same but switch `SCHEDULER` from `constant` to `linear`. What changes in the curve?
2) **Freeze the encoder**: freeze the base model and train only the classification head. Compare quality vs speed.
3) **Error analysis**: build a confusion matrix and inspect 5 misclassifications. Are there patterns in the mistakes?
4) **Experiment tracking (optional)**: log your run to Weights & Biases (W&B) *if your organisation/policies allow it*. See `handouts/Optional_WandB.md`.

---

## Links to Unit 6.2

This workshop is intentionally built around the Unit 6.2 concepts:

- loss functions: “measuring how wrong we are”
- gradient descent: “walking downhill”
- learning rate: “how big are your steps?”
- schedules: “adapting over time”
- why training costs time + money
