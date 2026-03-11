# Workshop 6W Cheat Sheet: Optimisation & Tuning (Unit 6)

## The one-sentence mental model

Training is the process of making a model **less wrong** by repeatedly nudging its
internal numbers (weights) using **gradients** to reduce **loss**. That is optimisation.

---

## The minimum maths vocabulary (so you can read papers and tooling)

| Term | What it means |
|---|---|
| **Loss** | A number that measures "how wrong" the model is. |
| **Differentiable** | Smooth enough that we can compute a slope. |
| **Gradient** | The slope of the loss — a set of partial derivatives, one per weight. |
| **SGD** | Taking downhill steps using gradients estimated from mini-batches. |
| **Learning rate** | Step size — how far we move in each update. |
| **Convergence** | The point where further steps produce negligible improvement. |
| **Generalisation** | How well the model works on unseen data. |
| **Generalisation gap** | The difference between training and validation performance. |

See `Unit6_Maths_Words_Glossary.md` for the full one-page version.

---

## Optimising vs fine-tuning vs retraining (plain English)

**Optimising** is the maths engine: reduce a loss function by adjusting weights.

**Fine-tuning** means running optimisation starting from a *pretrained* model
(a good starting point), using your domain data.

**Retraining** means running optimisation again because your data changed
(new data, drift, new labels). It might start from a pretrained model (common)
or from scratch (rare and expensive).

> The process is always optimisation. The difference is *where you start* and *why*.

---

## The core training loop

1. Predict (forward pass)
2. Measure wrongness (loss)
3. Compute the slope (gradients)
4. Take a step (update weights)
5. Repeat

---

## What to watch (metrics that actually help)

**Loss is altitude** — are we going downhill mathematically?

**Validation accuracy/F1 is the road test** — does it work on unseen data?

**Generalisation gap:**
- train improving + validation worsening → gap widening → likely overfitting

---

## Convergence: when to stop

A run has **converged** when the loss delta between epochs drops below a meaningful
threshold. At that point, further training is unlikely to produce improvement worth
the cost.

A run that has **not** converged is still moving — performance metrics taken mid-run
may not reflect deployment behaviour.

```
Simple check:
  delta = |loss_epoch_n - loss_epoch_(n-1)|
  if delta < 0.01:  # adjust threshold to your task
      # likely converged
```

Stopping early (for practical reasons) is a legitimate engineering decision — but
record it in your experiment log so the choice is auditable.

---

## Learning rate: the first dial to try

If something is wrong, the first dial to consider is almost always **learning rate**.

| Symptom | Most likely cause |
|---|---|
| Curve barely moves | LR too low |
| Curve oscillates or loss → NaN | LR too high |
| Smooth downward improvement | LR about right |

---

## Fixed vs decaying learning rate schedules

**Fixed (constant):** Same step size throughout. Simple to reason about.
Useful when you have limited compute and want predictable behaviour.

**Decaying (e.g. linear warmup then decay):** Large steps early, smaller steps later.
Often produces a better final result because it allows fine-grained adjustment as
the model approaches a minimum.

Trade-off: a decaying schedule adds a hyperparameter (the schedule shape) to tune.
For short fine-tuning runs, a well-chosen fixed rate often performs comparably.

> ⚠️ **In this workshop's notebook:** When you set `SCHEDULER = "linear"` (Activity 8 Option A), the notebook includes a **10% warmup phase** before the decay begins. This means the learning rate actually *increases* for the first 10% of training steps, then decays linearly to zero for the remainder. If you look at any diagnostic output showing per-step LR, you will see a small initial rise followed by a longer decline — this is expected and correct, not a bug. The warmup prevents instability in the very early steps when the model's gradients are noisiest.

---

## Compute cost and sustainability

Every training epoch consumes time, energy, and money. The threshold you choose for
"good enough" convergence is not only a mathematical decision — it is a business and
environmental one.

Practical "do not waste compute" rules:

- Always take a baseline before you change anything.
- Change one dial at a time.
- Stop training when you reach convergence, not when time runs out.
- Time-box experiments and document why you stopped.
- Consider: would a smaller model or a simpler approach meet the business need?

---

## Data quality matters for your conclusions

The metrics you get are only as trustworthy as the data you trained and validated on.

Questions to ask before claiming improvement:

- Is the validation set representative of real deployment data?
- Is the training data labelled consistently and accurately?
- Is my validation set large enough that a 2% accuracy difference is meaningful,
  rather than noise?
- If I used synthetic data, does it capture the distribution of the real task?

A modest improvement on a clean, representative baseline is more valuable than a
high number from a leaky or synthetic evaluation.

---

## Practical discipline (for engineering and for Task 6)

A well-kept experiment log is the difference between "I think the model improved"
and "I can demonstrate and explain the improvement". For Task 6 purposes, your log
is your S9 evidence.

Log at minimum:
- What you changed (one dial at a time)
- What you expected to happen
- What actually happened
- Your interpretation
- Whether the run converged
