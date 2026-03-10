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

These cells compute the **loss delta** between consecutive epochs — how much did loss change from one step to the next?

```
Epoch 2 → 3 loss delta: 0.041   ← still moving
Epoch 3 → 4 loss delta: 0.008   ← approaching convergence
Epoch 4 → 5 loss delta: 0.003   ← converged within threshold
```

If the delta falls below the threshold (default: `0.01`), the run is considered converged.

📘 **Convergence in plain English:** Think of a kettle approaching boiling point. The temperature is still rising, but each additional second adds less heat gain than the last. At some point, waiting longer gives you nothing more — you are just burning energy. A training run works the same way. The question is not whether to stop, but *when*.

✅ **Checkpoint:** The convergence check cell ran without error and produced a delta value or verdict.

---

## 📝 Task 2 — Record Your Result

Complete the convergence section of your experiment log (Part B of the Activity Worksheet):

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

💡 **If you're finding it abstract:** think about any professional process where there is a point of diminishing returns — where continuing past a certain point costs more than it gains. Training a model past convergence is that problem, with measurable numbers attached. If an example from your own work comes to mind naturally, use it. If not, the kettle analogy is sufficient.

🤔 **Reflect:** Your Task 6 K18 question asks you to relate mathematical principles to the design and use of a model. This question is that connection made explicit.

---

## 🚀 Extension

In the convergence check cell (Cell 19), the threshold is currently hardcoded:

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
