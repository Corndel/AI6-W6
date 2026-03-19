# Activity 5: Convergence Check

**Primary KSB:** K18 — Mathematical and statistical principles (optimisation and compute cost)

🎯 **Learning Objective:** Use the convergence check cells to determine whether your model's training run converged, and explain why stopping at convergence matters for cost and reliability

## 📋 Expected Outputs

- Convergence check cells run and result recorded in your experiment log
- Written answer to both reflection questions below
- K18 maths-to-technique link articulated in your own words

---

## 📝 Task 1 — Run the Convergence Check Cells

📓 **Notebook:** Find and run the **convergence check cells** (labelled `6b`).

These cells read the training log produced by your run and compute the **loss delta** between the last two epochs — how much did loss change from one step to the next?

With the default 2-epoch setting, your output will look like this:

```
==================================================
CONVERGENCE CHECK
==================================================
  Epoch 1 loss : 0.8420
  Epoch 2 loss     : 0.6310
  Delta (change)              : 0.2110

  ⚠  This run has NOT clearly converged.
     Training loss is still changing meaningfully.
     Practical question: is the remaining improvement worth
     the additional compute cost and time for this business goal?

Business framing:
  You ran 2 epoch(s) on 800 training examples.
  Consider: how would you justify this compute spend to a project manager?
  Would a shorter run have been 'good enough' for the goal?

Sustainability note:
  Every training epoch consumes energy. The threshold you choose for
  'good enough' convergence is also an environmental and cost decision,
  not just a mathematical one.
```

Or, if the delta is small enough:

```
  ✓  This run APPEARS TO HAVE CONVERGED within the time budget.
     Further epochs are unlikely to produce significant improvement.
     Stopping here is a defensible engineering decision.
```

The threshold is `0.01` — if `Delta (change)` is below that, the cell judges the run as converged. With only 2 epochs the delta will usually be quite large (the model is still moving), which is expected: it is an honest reflection of a short run, not a sign that something went wrong.

📘 **Convergence in plain English:** Think of charging a phone. The first 50% fills quickly — you can watch the bar move. The next 30% is noticeably slower. The last 20%, from 80% to 100%, takes almost as long as everything before it combined. The battery chemistry means each additional unit of charge is harder to push in than the last.

This is why public EV chargers routinely cut off at 80% — the cost of the final 20% in time, heat, and wear is not worth it for most journeys. The car is already good enough to use.

A training run works the same way. Early epochs produce large drops in loss. Later epochs produce smaller and smaller improvements. At some point, the marginal gain from another epoch is not worth the compute cost. The question is not whether to stop — it is *when good enough is good enough*, and that is an engineering decision, not a mathematical one.

✅ **Checkpoint:** The convergence check cell ran without error and produced a delta value or verdict.

---

## 📝 Task 2 — Record Your Result

Complete the convergence section of your experiment log (Part B of the [Activity Worksheet](../handouts/Activity_Worksheet.md)):

| | Your answer |
|---|---|
| Final epoch loss delta | |
| Did the run converge? (Y / N / Unclear) | |
| If not: would more epochs be worth the compute cost for this task? Why? | |

💡 **Interpreting unclear results:**
- One epoch = one data point. You cannot assess convergence from a single step.
- Erratic loss (common with `TOO_HIGH`) = "unclear" is a legitimate answer. Note what happened.

---

## 📝 Task 3 — Convergence and Cost

Work in groups of **2–4**. Answer both questions in 2–3 sentences each.

**Question 1 — in your own words:**
> "What does convergence mean mathematically, and how does the loss delta check make that definition operational?"

> _______________________________________________________________

**Question 2 — the engineering decision:**
> "How does knowing when to stop training connect to a real business decision — about cost, time, or sustainability?"

> _______________________________________________________________

💡 **If you're finding it abstract:** think about any professional process where there is a point of diminishing returns — where continuing past a certain point costs more than it gains. Training a model past convergence is that problem, with measurable numbers attached. If an example from your own work comes to mind naturally, use it. If not, the phone charging analogy is sufficient: fast early gains, slowing progress, a practical cutoff point that makes sense even before you reach the theoretical maximum.

🤔 **Reflect:** Your Task 6 K18 question asks you to relate mathematical principles to the design and use of a model. This question is that connection made explicit.

---

## 🚀 Extension

In the **convergence check cell** (labelled `## 6b) Did your training converge?`), the threshold is currently hardcoded:

```python
converged = final_delta < 0.01
```

Edit this value directly to make the threshold stricter:

```python
converged = final_delta < 0.005   # stricter
```

Re-run the cell. Does your run now count as converged?

- What would using a stricter threshold mean in practice?
- In a high-stakes deployment (medical, legal, financial), would you use a stricter or looser threshold — and why?

---

🎓 **Complete** — proceed to [Activity 6](activity-6_start.md)
