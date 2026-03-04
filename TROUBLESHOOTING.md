# Workshop 6W — Troubleshooting Guide

This document covers the most common failure modes and their fixes, organised by
environment type. Read the section for your delivery environment first.

---

## Quick reference: what broke and how to recover

| Symptom | Most likely cause | Fix |
|---|---|---|
| `train_test_split` fails with `TypeError: only integer scalar arrays` | pandas 3.x + backup notebook `.values` bug (now patched) | Update to the patched notebook; use `.to_numpy()` |
| `financial_phrasebank` download fails or hangs | HuggingFace blocked or dataset renamed | Fallback activates automatically; or use Plan B |
| DistilBERT download times out | Poor Wi-Fi / 20 simultaneous downloads | Pre-cache before the session; or use Plan B |
| `ModuleNotFoundError: No module named 'torch'` | PyTorch not installed | Run `pip install torch` or pre-install via requirements |
| Training cell crashes with `NaN loss` | `TOO_HIGH` preset on this hardware — expected | Record it; it is a teaching moment |
| `TRAINING_FAILED` is undefined | Cells run out of order | Restart kernel and run all cells top to bottom |
| Notebook hangs indefinitely on CPU | Machine is too slow | Reduce speed dials (see below) |
| `FileNotFoundError: synth dataset` | Jupyter launched from wrong directory | See directory fix below |
| `ValueError: eval_strategy` | transformers < 4.46 | Upgrade: `pip install "transformers>=4.46,<5"` |

---

## Speed dial reference (use these when training is too slow)

Open Cell 4 in the main notebook and reduce these values before running:

```python
TRAIN_SUBSET = 400   # default 800 — reduces training data
EPOCHS = 1           # default 2 — fastest possible run
MAX_LENGTH = 64      # default 96 — shorter sequences = faster tokenisation and training
```

On a typical modern CPU with these settings, one epoch should complete in 3–6 minutes.
If it is still slower than that, switch to Plan B.

---

## Plan B: when to switch and how

Switch to Plan B (`lab/02_backup_sgd_text_classifier_learning_rate.ipynb`) when:

- HuggingFace downloads are blocked or timing out after 5 minutes
- Training is too slow to complete within the session time even with minimum speed dials
- PyTorch cannot be installed in the environment

Plan B requires only: `numpy`, `pandas`, `scikit-learn`, `matplotlib` — all lightweight
and almost always pre-installed. It runs entirely offline using the synthetic CSV.

It teaches the same Unit 6 concepts (SGD, learning rate, convergence, generalisation gap)
and all worksheet tasks apply unchanged.

---

## Environment-specific guidance

### Pluralsight Cloud Sandbox — AWS

Pluralsight AWS sandboxes typically give you a temporary AWS account with console access.
The most practical environment for this workshop is an **EC2 instance running Amazon Linux
or Ubuntu**, accessed via the AWS console and a terminal session.

**Recommended instance type:** `t3.medium` (2 vCPU, 4 GB RAM) is sufficient for the
main notebook with reduced speed dials. `t3.large` (8 GB RAM) is more comfortable.
Avoid `t2.micro` — it does not have enough RAM to load DistilBERT.

**Step-by-step setup (AWS EC2):**

1. Launch an EC2 instance from the console:
   - AMI: Ubuntu 22.04 LTS (search "Ubuntu" in the AMI selector)
   - Instance type: `t3.medium` or larger
   - Security group: allow inbound SSH (port 22) from your IP; allow inbound port 8888
     for Jupyter (restrict to your IP)
   - Key pair: create or select one; download the `.pem` file

2. Connect via SSH:
   ```bash
   ssh -i your-key.pem ubuntu@<public-ip>
   ```

3. Install Python and dependencies:
   ```bash
   sudo apt update && sudo apt install -y python3-pip python3-venv git unzip
   python3 -m venv workshop_env
   source workshop_env/bin/activate
   pip install -r requirements.txt
   ```

4. Transfer workshop files (choose one method):
   - Upload via SCP: `scp -i your-key.pem AI6-6W.zip ubuntu@<public-ip>:~/`
   - Or clone from your organisation's Git repo if hosted there

5. Pre-cache the model (do this before the session):
   ```bash
   python3 -c "
   from transformers import AutoTokenizer, AutoModelForSequenceClassification
   AutoTokenizer.from_pretrained('distilbert-base-uncased')
   AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=3)
   print('Model cached successfully')
   "
   ```

6. Launch Jupyter:
   ```bash
   jupyter lab --ip=0.0.0.0 --port=8888 --no-browser
   ```
   Copy the token URL from the terminal and open it in a browser using the instance's
   public IP: `http://<public-ip>:8888/?token=<token>`

**AWS-specific notes:**

- Pluralsight sandbox accounts have a time limit (typically 2–4 hours). Do not rely on
  instance state persisting between sessions. Pre-install and pre-cache before learners arrive.
- Outbound HTTPS to `huggingface.co` and `pypi.org` is allowed by default in most
  sandbox configurations. If it is blocked, the fallback to the local CSV activates
  automatically, but DistilBERT cannot be loaded — switch to Plan B.
- The sandbox account cannot use SageMaker in all configurations. If SageMaker notebooks
  are available, they are also a valid option (see SageMaker section below).
- Do not store sensitive data or personal information in the sandbox — it is a temporary,
  shared environment.

**Using SageMaker notebooks (if available in your sandbox):**

1. Open SageMaker → Notebook Instances → Create
2. Instance type: `ml.t3.medium` is sufficient
3. Upload the workshop zip using the Jupyter file browser
4. Open a terminal in Jupyter and run `pip install -r requirements.txt`
5. The model cache location in SageMaker is `/home/ec2-user/SageMaker/` — files here
   persist across restarts of the same instance, unlike the general `/tmp/` area

---

### Pluralsight Cloud Sandbox — Azure

Pluralsight Azure sandboxes typically give you a subscription with a resource group.
The most practical options are an **Azure VM** or an **Azure Machine Learning compute instance**.

**Option 1: Azure VM (straightforward, no ML services required)**

1. In the Azure portal, create a Virtual Machine:
   - Image: Ubuntu 22.04 LTS
   - Size: `Standard_B2s` (2 vCPU, 4 GB RAM) minimum; `Standard_D2s_v3` preferred
   - Authentication: SSH public key or password
   - Inbound ports: allow SSH (22) and port 8888

2. Connect via SSH:
   ```bash
   ssh azureuser@<public-ip>
   ```

3. Install and set up (same steps as AWS EC2 above):
   ```bash
   sudo apt update && sudo apt install -y python3-pip python3-venv unzip
   python3 -m venv workshop_env
   source workshop_env/bin/activate
   pip install -r requirements.txt
   ```

4. Pre-cache the model and launch Jupyter (same as AWS EC2 above).

**Option 2: Azure Machine Learning compute instance (if available in the sandbox)**

1. Create an Azure ML workspace (if not already present)
2. Go to Compute → Compute Instances → New
3. Select `Standard_DS11_v2` (2 vCPU, 14 GB RAM) — this is the smallest instance with
   enough RAM for DistilBERT comfortably
4. Once running, click the Jupyter link in the portal
5. Open a terminal in Jupyter and run:
   ```bash
   pip install -r requirements.txt
   ```
6. Upload the workshop zip using the Jupyter file browser

**Azure-specific notes:**

- Azure sandbox subscriptions also have a time limit. The same pre-caching advice applies.
- Outbound HTTPS to `huggingface.co` is allowed from Azure VMs by default.
- If the sandbox restricts outbound internet (some enterprise sandbox configurations do),
  the local CSV fallback will be used for the dataset, but DistilBERT model downloads will
  fail — switch to Plan B.
- Azure ML compute instances have a managed conda environment. If you want to avoid
  dependency conflicts, create a new conda environment:
  ```bash
  conda create -n workshop6w python=3.11 -y
  conda activate workshop6w
  pip install -r requirements.txt
  ```

---

## Common error messages and fixes

### `TypeError: only integer scalar arrays can be converted to a scalar index`

**Where:** Backup notebook (Plan B), cell loading the dataset.

**Cause:** pandas 3.x returns Arrow-backed arrays in some configurations, and `.values`
does not produce a plain numpy array. The `train_test_split` `stratify` parameter fails
as a result.

**Fix:** The patched version of `02_backup_sgd_text_classifier_learning_rate.ipynb`
uses `.to_numpy()` instead of `.values`. If you are running an older copy of the notebook,
apply this change manually:

```python
# Replace this:
X_train, X_val, y_train, y_val = train_test_split(
    df["sentence"].values, df["y"].values,
    test_size=0.2, random_state=42, stratify=df["y"].values
)

# With this:
X_train, X_val, y_train, y_val = train_test_split(
    df["sentence"].to_numpy(), df["y"].to_numpy(),
    test_size=0.2, random_state=42, stratify=df["y"].to_numpy()
)
```

---

### `FileNotFoundError: Could not find synth dataset`

**Where:** Both notebooks, when trying to load the local CSV fallback.

**Cause:** Jupyter was launched from an unexpected directory. The notebook looks for the
CSV in two locations: `lab/data/synth_financial_sentiment.csv` (if launched from the repo
root) and `data/synth_financial_sentiment.csv` (if launched from inside `lab/`).

**Fix:** Check where Jupyter was launched from. In a Jupyter terminal cell, run:

```python
import os
print(os.getcwd())
```

Then confirm the CSV exists:

```bash
find . -name "synth_financial_sentiment.csv"
```

If necessary, restart Jupyter from the repo root directory:

```bash
cd /path/to/AI6-6W
jupyter lab
```

---

### `ModuleNotFoundError: No module named 'transformers'` (or torch, datasets, etc.)

**Cause:** Dependencies not installed, or the wrong Python environment is active.

**Fix:**

```bash
# Confirm which Python Jupyter is using
import sys; print(sys.executable)
```

Then install in that environment:

```bash
/path/to/python -m pip install -r requirements.txt
```

In a virtual environment, make sure it is activated before launching Jupyter:

```bash
source workshop_env/bin/activate
jupyter lab
```

---

### `ValueError: `eval_strategy` is not a valid field`

**Cause:** An older version of `transformers` is installed (below 4.41). The parameter
was renamed from `evaluation_strategy` to `eval_strategy` in 4.41.

**Fix:**

```bash
pip install "transformers>=4.46,<5" --upgrade
```

---

### `OSError: We couldn't connect to huggingface.co`

**Cause:** Outbound HTTPS is blocked in the sandbox, or the connection timed out.

**Fix:** The notebook will automatically fall back to the local synthetic CSV for the
dataset. However, DistilBERT model weights cannot be loaded offline.

If this happens:
1. Close the main notebook
2. Open `lab/02_backup_sgd_text_classifier_learning_rate.ipynb`
3. Continue the session with Plan B — all learning objectives remain achievable

If you anticipate this issue, pre-cache the model before the session (see the setup
instructions above for your environment).

---

### Training hangs and does not produce any output

**Cause:** Usually the machine does not have enough RAM to load DistilBERT, or the CPU
is very slow.

**Fix (step by step):**

1. First, reduce speed dials:
   ```python
   TRAIN_SUBSET = 300
   EPOCHS = 1
   MAX_LENGTH = 64
   ```

2. Restart the kernel (Kernel → Restart & Run All).

3. If it still hangs after 10 minutes, check available memory:
   ```bash
   free -h
   ```
   DistilBERT requires approximately 500 MB RAM on CPU. If available RAM is below 1.5 GB,
   switch to Plan B.

4. If the machine has less than 2 GB RAM, Plan B is the correct choice for the session.

---

### `NaN` loss or loss becomes very large immediately

**Cause:** This is the expected behaviour for the `TOO_HIGH` learning rate preset. The
training cell is designed to catch this exception and set `TRAINING_FAILED = True`.

**What to do:** Do not restart or change settings. Record the crash in the experiment log
(it happened at which epoch, what the last visible loss value was). This is a teaching
moment — a crashed `TOO_HIGH` run is as instructive as a clean `JUST_RIGHT` run.

If `NaN` loss happens on `JUST_RIGHT` or `TOO_LOW`, this is unexpected. Try:

```bash
# Restart the kernel and re-run from the top
# If it happens again, reduce MAX_LENGTH to 64 and TRAIN_SUBSET to 300
```

---

### `TRAINING_FAILED is not defined`

**Cause:** Cells were run out of order. `TRAINING_FAILED` is defined in the training
cell (Cell 17). If that cell has not run, subsequent cells that check it will fail.

**Fix:** Restart the kernel and run all cells in order from the top:
Kernel → Restart Kernel and Run All Cells.

---

### `TRAIN_EVAL_SUBSET` cell shows "not a representative estimate"

This is not an error — it is a comment in the notebook. The training subset evaluation
uses a maximum of 500 examples. For very small training subsets this is fine. The note
is there to remind learners that the generalisation gap estimate is approximate.

---

## Memory and compute reference

| Environment | RAM | DistilBERT main notebook | Plan B |
|---|---|---|---|
| `t2.micro` / `B1s` (1 GB) | 1 GB | ✗ Not enough | ✓ |
| `t3.medium` / `B2s` (4 GB) | 4 GB | ✓ With reduced speed dials | ✓ |
| `t3.large` / `D2s_v3` (8 GB) | 8 GB | ✓ Default settings | ✓ |
| `ml.t3.medium` SageMaker (4 GB) | 4 GB | ✓ With reduced speed dials | ✓ |
| `ml.c5.xlarge` SageMaker (8 GB) | 8 GB | ✓ Default settings | ✓ |
| Azure DS11_v2 (14 GB) | 14 GB | ✓ Default settings | ✓ |
| Standard laptop (8 GB+) | 8 GB | ✓ Default settings | ✓ |

GPU is not required. If available it will be used automatically and will make training
significantly faster, but the workshop is designed to run on CPU.

---

## Pre-session coach checklist

Run through this list at least 24 hours before the session, in the target environment:

- [ ] Can you install packages from PyPI? (`pip install numpy` as a test)
- [ ] Can you reach `huggingface.co`? (`curl -I https://huggingface.co`)
- [ ] Did the model pre-cache successfully? (run the cache command above)
- [ ] Does one epoch of training complete in under 8 minutes?
- [ ] Does the backup notebook run end to end? (`02_backup...ipynb`, all cells)
- [ ] Do learner accounts have access to the same environment you tested?
- [ ] Do you know where the Jupyter server URL / token will be for each learner?
- [ ] Have you prepared the group assignment slide (TOO_LOW / JUST_RIGHT / TOO_HIGH)?
