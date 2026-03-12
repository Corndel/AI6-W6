# Workshop 6W — Troubleshooting Guide

This document covers environment setup and the most common failure modes. If you are
setting up for the first time, **start with the section for your environment** under
"Environment-specific guidance" below — do not skip straight to the error entries.

---

## Quick reference: what broke and how to recover

| Symptom | Most likely cause | Fix |
|---|---|---|
| `train_test_split` fails with `TypeError: only integer scalar arrays` | pandas 3.x + backup notebook `.values` bug (now patched) | Update to the patched notebook; use `.to_numpy()` |
| `financial_phrasebank` download fails or hangs | HuggingFace blocked or dataset renamed | Fallback activates automatically; or use Plan B |
| DistilBERT download times out | Poor Wi-Fi / 20 simultaneous downloads | Pre-cache before the session; or use Plan B |
| `ModuleNotFoundError: No module named 'torch'` | PyTorch not installed | Run `pip install -r requirements.txt` or pre-install |
| Training cell crashes with `NaN loss` | `TOO_HIGH` preset on this hardware — expected | Record it; it is a teaching moment |
| `TRAINING_FAILED` is undefined | Cells run out of order | Restart kernel and run all cells top to bottom |
| Notebook hangs indefinitely on CPU | Machine is too slow | Reduce speed dials (see below) |
| `FileNotFoundError: synth dataset` | Jupyter launched from wrong directory | See directory fix below |
| `ValueError: eval_strategy` | transformers < 4.46 | Upgrade: `pip install "transformers>=4.46,<5"` |
| SSH refuses key with "Permissions too open" | `.pem` file needs restricted permissions | Run `chmod 400 your-key.pem` first |
| Jupyter token URL shows `localhost` but won't open | Terminal shows localhost; you need the public IP | Replace `localhost` with the EC2 public IP |

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

If you are in an environment where `pip install -r requirements.txt` fails because
`torch` cannot be installed (firewall, no binary available), install only what Plan B
needs:
```bash
pip install numpy pandas scikit-learn matplotlib
```

It teaches the same Unit 6 concepts (SGD, learning rate, convergence, generalisation gap)
and all worksheet tasks apply unchanged.

### ⚠️ Label noise in Plan B — read this before delivering

The backup notebook runs with `NOISE_FRACTION = 0.25` active by default. This means that
before any training happens, **25% of the training labels are randomly corrupted** — a
quarter of examples get a wrong label assigned.

**Why it is there:** The noise makes the differences between TOO_LOW, JUST_RIGHT, and
TOO_HIGH much sharper and easier to see. Without noise, the three presets can converge
towards similar final accuracy; with noise, the instability of TOO_HIGH becomes much more
visible and the pedagogical contrast is clearer.

**What it changes for you:**

- **Accuracy ceiling:** No matter how well the learning rate is tuned, Plan B will not
  produce validation accuracy much above **70–75%** on this 3-class problem. That is not
  a bug — it is the mathematical consequence of 25% label noise. A perfect model cannot
  score above approximately 75% when a quarter of its training labels are wrong.

- **Comparing results across the room:** If some learners are on the main notebook and
  some are on Plan B, their absolute accuracy figures are not directly comparable. During
  the Activity 4 share-back, make this explicit: *"Plan B results have a lower ceiling
  because the data has intentional noise. The shape of the curve and the pattern across
  presets still tell the same story."*

- **Teaching opportunity in Activity 6:** You can use this directly in the data quality
  discussion: *"Your training data for Plan B had 25% label noise baked in. That is a
  controlled version of something that happens in real annotation pipelines. How did that
  affect what you could conclude from your results?"*

**How to disable the noise:** Open Cell 2 and change `NOISE_FRACTION = 0.25` to
`NOISE_FRACTION = 0.0`. This produces a clean run. Learning rate effects will still be
visible, but the contrast between presets is less dramatic.

---

## Environment-specific guidance

The sections below cover every supported environment. **GitHub Codespaces is recommended
as the lowest-friction option** for most learners — it requires only a browser and a
GitHub account. If your cohort is using Pluralsight sandboxes, use the AWS or Azure
section. If internet access is restricted on the day, use Plan B.

---

### GitHub Codespaces (recommended — fastest to start)

Codespaces gives every learner a fully configured Linux environment in their browser with
no local installation, no SSH, no firewall configuration, and no security groups. Because
the repo is public, any learner with a free GitHub account can be running in under
2 minutes.

**Setup:**

The repo includes a `.devcontainer/devcontainer.json` that configures Python 3.11 and
runs `pip install -r requirements.txt` automatically when the Codespace is created.
This means there is no manual install step — just click and wait.

1. Go to [https://github.com/Corndel/AI6-W6](https://github.com/Corndel/AI6-W6) in a browser.
2. Click the green **Code** button → **Codespaces** tab → **Create codespace on main**.
3. Wait for the environment to build (typically 3–5 minutes on first launch — it is
   installing all dependencies automatically in the background). You will see a VS Code
   interface appear; wait for the terminal at the bottom to show "postCreateCommand"
   completing before opening the notebook.
4. In the file browser (left sidebar), navigate to `lab/` and open
   `01_finetune_distilbert_optimisation_in_the_wild.ipynb`.
5. When VS Code asks you to select a kernel, choose the Python option labelled something
   like **`Python 3.11.x /usr/local/bin/python`**. If prompted to install the Jupyter
   extension, accept. If you see "No kernel found", click **Select Another Kernel →
   Python Environments** and choose the `/usr/local/bin/python` entry.

That is it. The repo files are already present and all packages are pre-installed.

> **If the postCreateCommand fails** (visible in the Codespace terminal as a pip error):
> open a terminal (Terminal → New Terminal) and run `pip install -r requirements.txt`
> manually.

**Codespaces-specific notes:**

- **Do you need a GitHub account?** Yes. A free account at github.com is sufficient.
  If learners do not have one, account creation takes about 2 minutes. Alternatively,
  use Google Colab (see below) — it requires a Google account instead.

- **What if Codespaces is disabled?** Some organisations block Codespaces at the
  GitHub organisation level. If the Codespaces tab does not appear, use Colab or a
  local setup instead.

- **Cost and limits:** GitHub Free accounts include 120 core-hours per month. A 2-core
  Codespace running for a 5-hour workshop uses 10 core-hours — well within the free tier.
  Learners should **stop** (not just close) their Codespace after the session to avoid
  background usage: go to [github.com/codespaces](https://github.com/codespaces) and
  click **Stop**. Closing the browser tab does not stop the Codespace.

- **File persistence:** Files saved inside a Codespace persist across sessions, unlike
  Colab. Learners can close the browser and return to the same environment later.
  However, Codespaces are deleted after 30 days of inactivity — ask learners to download
  their worksheet before then: right-click the file in the sidebar → **Download**.

- **Model caching:** HuggingFace model weights cache normally inside the Codespace and
  persist between sessions on the same instance. No pre-caching step is needed.

- **Training speed:** The default Codespace uses a 2-core CPU. Training with the default
  speed dials (800 examples, 2 epochs) will take around 8–12 minutes per run. If this is
  too slow, reduce the speed dials (see the Speed dial reference above). GPU Codespaces
  are available but require a paid GitHub plan.

- **W&B (Activity 8 Option D):** Works normally. Run `wandb login` in the terminal and
  paste your API key when prompted.

---

### Google Colab

Colab is a good alternative when learners do not have a GitHub account or Codespaces is
unavailable. Google's network can reach `huggingface.co` reliably, and there is nothing
to install locally.

**Setup:**

1. Go to [colab.research.google.com](https://colab.research.google.com) and sign in with a Google account.
2. Create a new notebook: **File → New notebook**.
3. Run the following cells in order — do not open the workshop notebook first.

> ⚠️ Do not use **File → Open notebook → GitHub** to open the workshop notebook directly.
> If you do, the dependencies will not be installed and all imports will fail. Follow the
> steps below instead.

```python
# Cell 1 — clone the repo
!git clone https://github.com/Corndel/AI6-W6.git
%cd AI6-W6
```

```python
# Cell 2 — install dependencies (takes 2–3 minutes — wait for it to finish)
!pip install -r requirements.txt -q
```

Then navigate in the Colab file browser (folder icon, left sidebar) to
`AI6-W6/lab/01_finetune_distilbert_optimisation_in_the_wild.ipynb` and click to open it.
It will open in the same runtime session, so your installed packages will be available.

**Colab-specific notes:**

- **Runtime type:** The default CPU runtime is sufficient. If GPU is available (Runtime →
  Change runtime type → T4 GPU), training will be noticeably faster, but the learning
  objectives are fully achievable on CPU.

- **Session timeouts:** Colab sessions time out after 90 minutes of inactivity and
  after a maximum of 12 hours (free tier). If a learner's session disconnects mid-training,
  they will lose their kernel state and need to re-run all cells from the top. Pre-warn
  learners to keep their browser tab active and not to leave it idle.

- **File persistence:** Files saved to the Colab runtime (`/content/AI6-W6/`) are lost
  when the session ends. Ask learners to download their worksheet before closing:
  right-click the file in the file browser → **Download**.

- **Model caching:** HuggingFace model weights download to `/root/.cache/huggingface/`
  on Colab. This cache is lost when the session ends — each new session requires a fresh
  download (~250 MB). On Google's network this is fast, but pre-warn learners if they are
  on a slow connection.

- **W&B (Activity 8 Option D):** Works on Colab. The `wandb login` step requires pasting
  an API key in the notebook output area — straightforward and works the same as local.

---

### Pluralsight Cloud Sandbox — AWS

Pluralsight AWS sandboxes give you a temporary AWS account with console access.
The most practical environment for this workshop is an **EC2 instance running Ubuntu**,
accessed via a terminal session.

> ⚠️ **Sandbox time limits:** Pluralsight sandbox accounts typically expire after 2–4 hours.
> This workshop runs for 5 hours. Either use a sandbox with a longer limit, split the
> session across two sandbox activations, or switch to Codespaces/Colab instead.

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

2. Fix the key file permissions and connect via SSH:

   **Mac / Linux:**
   ```bash
   chmod 400 your-key.pem
   ssh -i your-key.pem ubuntu@<public-ip>
   ```
   > If you skip `chmod 400`, SSH will refuse the key with a "Permissions too open" error.

   **Windows:** Use [Windows Subsystem for Linux (WSL)](https://learn.microsoft.com/en-us/windows/wsl/install)
   and run the same commands above, or use [PuTTY](https://www.putty.org/) with the `.pem`
   converted to `.ppk` format using PuTTYgen.

3. Install system packages:
   ```bash
   sudo apt update && sudo apt install -y python3-pip python3-venv git unzip
   ```

4. Get the workshop files (choose one method):

   **Option A — clone from GitHub (recommended, requires outbound internet):**
   ```bash
   git clone https://github.com/Corndel/AI6-W6.git
   cd AI6-W6
   ```

   **Option B — upload a zip from your local machine:**
   From a terminal *on your local machine* (a second terminal, not the SSH session), run:
   ```bash
   scp -i your-key.pem AI6-W6-main.zip ubuntu@<public-ip>:~/
   ```
   Then back in the SSH session, unzip and enter the directory:
   ```bash
   unzip AI6-W6-main.zip
   cd AI6-W6-main
   ```

5. Create a virtual environment and install dependencies:
   ```bash
   python3 -m venv workshop_env
   source workshop_env/bin/activate
   pip install -r requirements.txt
   ```

   > The virtual environment must remain active for steps 6 and 7. If you open a new
   > SSH terminal at any point, re-activate it before continuing:
   > `source workshop_env/bin/activate` (run this from inside the repo directory).

6. Pre-cache the model (do this before the session — requires outbound HTTPS):
   ```bash
   python3 -c "
   from transformers import AutoTokenizer, AutoModelForSequenceClassification
   AutoTokenizer.from_pretrained('distilbert-base-uncased')
   AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=3)
   print('Model cached successfully')
   "
   ```

7. Launch Jupyter:
   ```bash
   jupyter lab --ip=0.0.0.0 --port=8888 --no-browser
   ```
   The terminal will print a URL that looks like:
   ```
   http://localhost:8888/?token=abc123...
   ```
   > ⚠️ Do **not** use `localhost` — that refers to the EC2 instance, not your machine.
   > Replace `localhost` with the instance's **public IP address**:
   > `http://<public-ip>:8888/?token=abc123...`

**AWS-specific notes:**

- Outbound HTTPS to `huggingface.co` and `pypi.org` is allowed by default in most
  sandbox configurations. If blocked, the fallback to the local CSV activates
  automatically for the dataset, but DistilBERT cannot be loaded — switch to Plan B.
- The sandbox account cannot use SageMaker in all configurations. If SageMaker notebooks
  are available, they are also a valid option (see SageMaker section below).
- Do not store sensitive data or personal information in the sandbox — it is a temporary,
  shared environment.

**Using SageMaker notebooks (if available in your sandbox):**

1. Open SageMaker → Notebook Instances → Create
2. Instance type: `ml.t3.medium` is sufficient
3. Once running, click **Open JupyterLab**
4. Open a terminal inside JupyterLab (File → New → Terminal) and clone the repo:
   ```bash
   cd ~/SageMaker
   git clone https://github.com/Corndel/AI6-W6.git
   cd AI6-W6
   pip install -r requirements.txt
   ```
   Or, if cloning is blocked, upload the zip via the JupyterLab file browser, then in
   the terminal:
   ```bash
   cd ~/SageMaker
   unzip AI6-W6-main.zip
   cd AI6-W6-main
   pip install -r requirements.txt
   ```
5. Open `lab/01_finetune_distilbert_optimisation_in_the_wild.ipynb` from the file browser.
   When prompted to select a kernel, choose **conda_python3** — this is the environment
   where packages installed with plain `pip` in the terminal land on SageMaker Notebook
   Instances. If you see a "kernel not found" prompt when opening the notebook, select
   `conda_python3` from the dropdown.
6. The model cache lives at `/home/ec2-user/SageMaker/` on Notebook Instances — files
   here persist across restarts of the same instance, so pre-caching once is sufficient.

---

### Pluralsight Cloud Sandbox — Azure

Pluralsight Azure sandboxes give you a subscription with a resource group. The most
practical options are an **Azure VM** or an **Azure Machine Learning compute instance**.

> ⚠️ **Sandbox time limits:** Same caveat as AWS — check your sandbox expiry against
> the 5-hour session length before the day.

**Option 1: Azure VM (straightforward, no ML services required)**

1. In the Azure portal, create a Virtual Machine:
   - Image: Ubuntu 22.04 LTS
   - Size: `Standard_B2s` (2 vCPU, 4 GB RAM) minimum; `Standard_D2s_v3` preferred
   - Authentication: SSH public key or password
   - Inbound ports: allow SSH (22) and port 8888

2. Fix permissions and connect via SSH (same `chmod 400` caveat as AWS applies):

   **Mac / Linux:**
   ```bash
   chmod 400 your-key.pem
   ssh azureuser@<public-ip>
   ```
   **Windows:** Use WSL or PuTTY (same as AWS above).

3. Install system packages:
   ```bash
   sudo apt update && sudo apt install -y python3-pip python3-venv git unzip
   ```

4. Get the workshop files:
   ```bash
   git clone https://github.com/Corndel/AI6-W6.git
   cd AI6-W6
   ```
   If cloning is blocked, use SCP from your local machine (same as AWS Option B above),
   then `unzip AI6-W6-main.zip && cd AI6-W6-main`.

5. Create a virtual environment and install dependencies:
   ```bash
   python3 -m venv workshop_env
   source workshop_env/bin/activate
   pip install -r requirements.txt
   ```

   > Same note as AWS: if you open a new SSH terminal, re-activate with
   > `source workshop_env/bin/activate` before running the pre-cache command or Jupyter.

6. Pre-cache the model (same command as AWS step 6 above).

7. Launch Jupyter:
   ```bash
   jupyter lab --ip=0.0.0.0 --port=8888 --no-browser
   ```
   Same as AWS: replace `localhost` in the printed token URL with the VM's public IP.

**Option 2: Azure Machine Learning compute instance (if available in the sandbox)**

1. Create an Azure ML workspace (if not already present)
2. Go to Compute → Compute Instances → New
3. Select `Standard_DS11_v2` (2 vCPU, 14 GB RAM) — smallest comfortable option for
   DistilBERT
4. Once running, click **JupyterLab** in the portal
5. Open a terminal (File → New → Terminal) and get the repo:
   ```bash
   cd ~/cloudfiles/code
   git clone https://github.com/Corndel/AI6-W6.git
   cd AI6-W6
   ```
   If cloning is blocked, upload the zip via the JupyterLab file browser, then:
   ```bash
   cd ~/cloudfiles/code
   unzip AI6-W6-main.zip
   cd AI6-W6-main
   ```
6. Install dependencies into a dedicated conda environment (required — bare `pip install`
   will fail on newer Python with "externally-managed-environment"):
   ```bash
   conda create -n workshop6w python=3.11 -y
   conda activate workshop6w
   pip install -r requirements.txt
   ```
7. Open `lab/01_finetune_distilbert_optimisation_in_the_wild.ipynb` from the file browser.
   When prompted to select a kernel, choose **workshop6w** from the list. If it does not
   appear, refresh the kernel list or restart JupyterLab.

**Azure-specific notes:**

- Outbound HTTPS to `huggingface.co` is allowed from Azure VMs by default. If the
  sandbox restricts outbound internet, switch to Plan B.

---

## Error entries

---

### `TypeError: only integer scalar arrays can be converted to a scalar index`

**Where:** Backup notebook (Plan B), cell loading the dataset.

**Cause:** pandas 3.x returns Arrow-backed arrays in some configurations, and `.values`
does not produce a plain numpy array. The `train_test_split` `stratify` parameter fails
as a result.

**Fix:** The patched version of `02_backup_sgd_text_classifier_learning_rate.ipynb`
uses `.to_numpy()` instead of `.values`. If you are running an older copy, apply this
change manually:

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

**Fix:** Check where Jupyter was launched from. In a notebook cell, run:

```python
import os
print(os.getcwd())
```

Then confirm the CSV exists:

```bash
find . -name "synth_financial_sentiment.csv"
```

If necessary, restart Jupyter from the repo root directory. The correct path depends on
your environment:

| Environment | Repo root path |
|---|---|
| EC2 / Azure VM | `~/AI6-W6` (or `~/AI6-W6-main` if you used the zip) |
| Codespaces | `/workspaces/AI6-W6` |
| Azure ML | `~/cloudfiles/code/AI6-W6` |
| Local laptop | wherever you cloned it — check with `pwd` |

```bash
cd <repo-root>
jupyter lab
```

---

### `ModuleNotFoundError: No module named 'transformers'` (or torch, datasets, etc.)

**Cause:** Dependencies not installed, or the wrong Python environment is active.

**Fix:**

**In Codespaces:** There is no virtual environment — packages install directly to the
system Python. If imports fail, check that pip install completed successfully:
```bash
pip show transformers
```
If not installed, re-run `pip install -r requirements.txt` in the terminal, then
restart the Jupyter kernel (Kernel → Restart).

**On EC2, Azure VM, or local laptop:** First confirm which Python Jupyter is using:

```python
import sys; print(sys.executable)
```

Then install into that environment directly:

```bash
# Example: if sys.executable shows /home/ubuntu/AI6-W6/workshop_env/bin/python
/home/ubuntu/AI6-W6/workshop_env/bin/python -m pip install -r requirements.txt
```

In a virtual environment, make sure it is activated before launching Jupyter:

```bash
source workshop_env/bin/activate
jupyter lab
```

(This does not apply to Codespaces, which uses system Python with no venv.)

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
instructions for your environment above).

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
(at which epoch it happened, what the last visible loss value was). This is a teaching
moment — a crashed `TOO_HIGH` run is as instructive as a clean `JUST_RIGHT` run.

If `NaN` loss happens on `JUST_RIGHT` or `TOO_LOW`, this is unexpected. Try restarting
the kernel and re-running from the top. If it recurs, reduce `MAX_LENGTH` to 64 and
`TRAIN_SUBSET` to 300.

---

### `TRAINING_FAILED is not defined`

**Cause:** Cells were run out of order. `TRAINING_FAILED` is defined in the training
cell (Cell 17). If that cell has not run, subsequent cells that check it will fail.

**Fix:** Restart the kernel and run all cells in order from the top:
Kernel → Restart Kernel and Run All Cells.

---

### `TRAIN_EVAL_SUBSET` cell shows "not a representative estimate"

This is not an error — it is a comment in the notebook. The training subset evaluation
uses a maximum of 500 examples. The note is there to remind learners that the
generalisation gap estimate is approximate.

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
| GitHub Codespaces (2-core default) | ~8 GB | ✓ With reduced speed dials | ✓ |
| Standard laptop (8 GB+) | 8 GB | ✓ Default settings | ✓ |

GPU is not required. If available it will be used automatically and will make training
significantly faster, but the workshop is designed to run on CPU.

---

## Pre-session coach checklist

Run through this list at least 24 hours before the session, in the target environment:

- [ ] Have you chosen your delivery environment (Codespaces / Colab / EC2 / Azure)?
- [ ] If using Codespaces: have you confirmed learners have GitHub accounts, and that Codespaces is not disabled by their organisation?
- [ ] If using EC2/Azure: does the sandbox time limit cover the full 5-hour session?
- [ ] Can you install packages from PyPI? (`pip install numpy` as a test)
- [ ] Can you reach `huggingface.co`? (`curl -I https://huggingface.co`)
- [ ] Did the model pre-cache successfully? (run the cache command above)
- [ ] Does one epoch of training complete in under 8 minutes?
- [ ] Does the backup notebook run end to end? (`02_backup...ipynb`, all cells)
- [ ] Do learner accounts have access to the same environment you tested?
- [ ] Do you know how learners will access Jupyter (token URL, Codespace link, etc.)?
- [ ] Have you prepared the group assignment slide (TOO_LOW / JUST_RIGHT / TOO_HIGH)?
- [ ] Have you told learners to download their worksheet before closing the session?
