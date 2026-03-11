# Activity 2: Environment Setup & Baseline

**Primary KSB:** S9 — Refine or re-engineer models and pipelines

🎯 **Learning Objective:** Set up the workshop environment and record a pre-training baseline so you have a meaningful "before" measurement for Task 6

## 📋 Expected Outputs

- Virtual environment created and dependencies installed
- Jupyter Lab running with the main notebook open
- Baseline metrics recorded in your experiment log (Part B of the Activity Worksheet)

---

## 📝 Task 1 — Clone and Set Up

> If your coach has pre-provisioned a cloud sandbox, follow their instructions for accessing it. Otherwise, follow the steps below for your environment.

**Option A — Local laptop or cloud VM (recommended)**

⌨️ **Terminal:**

```bash
git clone https://github.com/Corndel/AI6-W6.git
cd AI6-W6
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows PowerShell
pip install -r requirements.txt
```

**Option B — Google Colab**

In a new Colab notebook, run these cells in order:

```python
# Cell 1 — clone the repo
!git clone https://github.com/Corndel/AI6-W6.git
%cd AI6-W6
```

```python
# Cell 2 — install dependencies
!pip install -r requirements.txt -q
```

Then open `lab/01_finetune_distilbert_optimisation_in_the_wild.ipynb` from the Colab file browser (folder icon on the left). Skip the `jupyter lab` step — Colab is already running a notebook environment. Note: model downloads on Colab use Google's network, which is generally fast and unblocked, so the HuggingFace fallback to the synthetic CSV is unlikely to activate.

✅ **Checkpoint:** No errors during `pip install`. You should see packages including `transformers`, `torch`, and `datasets` in the output.

⚠️ **Warning:** If you are in a Pluralsight sandbox and package downloads are slow or blocked, tell your coach now — switching to Plan B early costs less time than waiting 20 minutes.

> 📋 **If your coach switches to Plan B:** The backup notebook (`02_backup_sgd_text_classifier_learning_rate.ipynb`) teaches the same Unit 6 concepts using a simpler model that runs without internet. Everything in this activity scaffold still applies. One thing to be aware of: Plan B adds intentional label noise to the training data by default (25% of examples get the wrong label). This caps achievable accuracy at around 75% — not because the model is doing badly, but because no model can learn perfectly from noisy labels. Your coach will explain what this means for how to interpret your results, and it connects directly to the data quality discussion later in the day.

---

## 📝 Task 2 — Launch Jupyter and Open the Notebook

⌨️ **Terminal:**

```bash
jupyter lab
```

📓 **Notebook:** Open `lab/01_finetune_distilbert_optimisation_in_the_wild.ipynb`

Run **Cell 1** (installs) and **Cell 2** (imports). Wait for both to complete without errors.

✅ **Checkpoint:** Cell 2 completes with no `ImportError`. You should see a message confirming the DistilBERT tokeniser loaded (or a cached-model confirmation).

💡 **Tip:** If you see a message like `Some weights of DistilBertForSequenceClassification were not initialized...` — that is expected. The classification head is randomly initialised before fine-tuning.

---

## 📝 Task 3 — Record Your Baseline

Before you change anything, run the **baseline evaluation cells**.

These cells evaluate the model *before* any fine-tuning. Record the results in your **experiment log** (Part B of the Activity Worksheet):

| Field | Your value |
|---|---|
| Baseline accuracy | |
| Baseline F1-macro | |
| Notes | Untrained classification head |

📘 **Why this matters:** Without a baseline, you cannot claim improvement. Task 6 requires a before-and-after comparison — this is your "before". An untrained head should perform around random chance for 3 classes (~33% accuracy). If your baseline is already high, check with your coach.

✅ **Checkpoint:** Baseline metrics are recorded in your experiment log before you proceed to the training cells.

---

## 📝 Task 4 — Check Your Speed Dials

Find the **speed dial cells** near the top of the notebook. They look like:

```python
TRAIN_SUBSET = 800
MAX_LENGTH   = 96
EPOCHS       = 2
```

💡 **Tip — if training is going to be slow:**

| Setting | Default | Reduce to |
|---|---|---|
| `TRAIN_SUBSET` | 800 | 300–400 |
| `MAX_LENGTH` | 96 | 64 |
| `EPOCHS` | 2 | 1 |

Reducing these does **not** undermine the learning — you are still observing real optimisation. It just completes faster.

✅ **Checkpoint:** You know where the speed dials are and have adjusted them if needed. Do not start training yet — wait for your group assignment in Activity 3.

---

## 🚀 Extension

If you finish setup early, open `lab/02_backup_sgd_text_classifier_learning_rate.ipynb` and read the first two markdown cells.

Then think back to the "You've seen this before" slide from the opening session. Plan B uses SGDClassifier — a simpler model, but the same underlying mechanism: iterative steps downhill, a learning rate controlling step size, a validation signal telling you whether it is working.

- Which of the examples on that slide maps most cleanly onto what Plan B is doing?
- What does "stochastic" mean in that context — and why does using mini-batches rather than the full dataset actually help rather than hinder?

No need to write a formal answer. This is for your own understanding before Activity 3.

---

🎓 **Complete** — proceed to [Activity 3](activity-3_start.md)
