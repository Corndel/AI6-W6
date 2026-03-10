# Activity 4: Share & Compare

**Primary KSBs:** B3 — Acts inclusively when collaborating with people from technical and non-technical backgrounds. Contributing to knowledge sharing, management and empowerment across the broader team; K18 — Mathematical and statistical principles

🎯 **Learning Objective:** Synthesise results across groups to identify which learning rate produced the healthiest training run, and practise explaining findings to a non-technical audience

## 📋 Expected Outputs

- Your group's metrics entered into the shared results slide (or board)
- Written answer to the engineering conclusion question
- One-paragraph stakeholder summary drafted

---

## 📝 Task 1 — Report Into the Shared Slide

Your coach will display (or share) a results slide with columns:

**Preset | LR value | Train loss (end) | Val loss (end) | Accuracy | F1-macro | Generalisation gap | Notes**

Enter your group's values from your experiment log — do not estimate from memory.

✅ **Checkpoint:** All groups have entered their results before the discussion begins.

---

## 📝 Task 2 — Whole-Group Discussion

Your coach will facilitate. Take notes — the observations from other groups feed directly into your Task 6 reflection.

**1. Compare the curves.**
Did the results match the predictions your groups made before running? Where were the surprises?

**2. Compare the generalisation gaps.**
Which preset produced the healthiest gap? Which overfitted? Which barely moved?

> _______________________________________________________________

**3. Engineering conclusion.**
If you had to recommend one preset for a production run based only on today's evidence, which would you choose — and what would you want to know before committing?

> My recommendation: _______________________________________________________________

> What I would want to know first: _______________________________________________________________

---

## 📝 Task 3 — What Patterns Came Up? (3 minutes)

During the warm-up and the run, some of you may have noticed connections to things you work on outside ML. Take 3 minutes to share briefly — not to construct a formal analogy, just to notice.

> "In Activity 1 we saw examples of these patterns outside ML. Did anything from your run remind you of one of those — or of something from your own work?"

💡 There is no right answer and no obligation. If nothing came up, that is fine. If something did, the whole group benefits from hearing it — people working in different sectors often recognise the same dynamics under different names.

📘 **Why this matters (B3):** Cross-sector knowledge sharing is not a soft extra. When an ML engineer can say "this is similar to how a physiotherapy programme calibrates load progression", they build bridges between technical and non-technical colleagues. That is what B3 is about: empowering the broader team, not just reporting upwards.

---

## 📝 Task 4 — Stakeholder Framing (B3)

🤔 **Scenario:** The person who approved the compute budget for this experiment wants a one-paragraph update. They understand your organisation but not machine learning.

Write 3–5 sentences. Cover: what you did, what you found, what it means for the next step, and any caveats. Translate the result — do not just report the numbers.

> _______________________________________________________________

💡 **Tip:** Swap "loss" for something they would recognise. "How wrong the predictions were." "How close the model got." What does *better* look like to them — fewer errors on customer queries, more accurate flags, fewer manual reviews? That is the language.

---

## 🚀 Extension

Look at the generalisation gap across all three presets. Is there a pattern between step size and gap size? Write two sentences: one for a technical colleague, one for a non-technical one. What do you have to change between the two versions?

---

🎓 **Complete** — proceed to [Activity 5](../activity-5/activity-5_start.md)
