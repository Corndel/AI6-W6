# Tuning Decision Log (Template)

Use this when you want your tuning choices to be **explainable** (to your coach, your team, or an auditor).

## 1) Context

- Model / approach:
- Task:
- Dataset (where from?):
- Train/validation split method:
- Compute / time budget:

## 2) Baseline

- Baseline config (what did you start from?):
- Baseline metrics (val loss, accuracy, F1):
- Baseline note (what surprised you?):

## 3) Experiments you ran (one dial at a time)

| Run | What dial changed? | Setting | Expected behaviour | Observed behaviour | Key metric change |
|---|---|---|---|---|---|
| 1 |  |  |  |  |  |
| 2 |  |  |  |  |  |
| 3 |  |  |  |  |  |

## 4) Decision

- Chosen configuration:
- Why this choice is “good enough” for the goal:
- What you deliberately did *not* optimise (and why):

## 5) Evidence

- Training curve summary (what did it look like?):
- Validation behaviour (generalisation):
- Any risks noticed (overfitting, instability, data issues):

## 6) Next steps (if you had more time)

- One data improvement you would try:
- One modelling/tuning improvement you would try:
- One evaluation improvement you would try:
