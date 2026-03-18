# Activity 6: Data Quality & Uncertainty

**Primary KSBs:** B5 — Data quality and working under uncertainty; S14 — Statistical analysis

🎯 **Learning Objective:** Identify the key limitations of today's data and practise framing those limitations honestly for a non-technical audience

## 📋 Expected Outputs

- Two data quality limitations identified and written down
- Stakeholder caveats drafted
- B5 reflection notes ready to feed into Task 6

---

## 📝 Task 1 — Where Did This Data Come From?

Take 60 seconds to discuss in groups of **2–4**:

> "What do you actually know about the data you trained on today — and what do you not know?"

📘 **The data, explained:**

**`financial_phrasebank` (`sentences_allagree` subset):** Only the cases where all human annotators agreed on the label — the unambiguous slice. Real-world financial text is rarely this clean.

**`synth_financial_sentiment.csv`:** Synthetic — generated rather than collected. A model trained only on it would require further validation before production use.

**Training subset size:** 300–800 examples. A 2% accuracy difference on a validation set this small may be noise, not a genuine improvement. "Statistically significant" requires knowing the margin of error.

---

## 📝 Task 2 — Identify Two Limitations

Write down the two most important data quality limitations from today. Be specific — not "the data was small" but *why* the smallness matters for the conclusions you drew.

**Limitation 1:**

> _______________________________________________________________

**Limitation 2:**

> _______________________________________________________________

---

## 📝 Task 3 — Deployment Question (B5)

🤔 **Scenario:** Someone in your organisation asks whether the model you trained today is ready to deploy next month.

Write a response of 3–5 sentences. Cover: what the results show, what they do not show, and what you would need before recommending deployment.

> _______________________________________________________________

💡 **Prompt:** Think about what a false positive or a false negative would cost in the context where this model might be used. A model with 70% accuracy can be perfectly acceptable in one context and completely unacceptable in another. That context — and that cost — is what your response needs to communicate.

💡 **Grounding it:** If a concrete deployment scenario from your own work comes to mind naturally, use it as the frame. If not, pick any of these: fraud detection, clinical triage, customer churn prediction, content moderation. The reasoning about data limitations and deployment risk transfers across all of them.

📘 **B5 note:** Honest caveats are professional, not a sign of failure. An overconfident claim about a model trained on 600 synthetic examples is a more serious problem than a modest result stated clearly.

---

## 📝 Task 4 — B5 Link for Task 6

Complete this sentence:

> "The data constraints I encountered in this workshop affected my conclusions because ___, and before treating these results as production-ready I would need ___."

> _______________________________________________________________

---

## 🚀 Extension

Think about a dataset you have actually worked with — or one that is publicly known in your field.

- What would the equivalent of the `allagree` subset look like — the high-confidence, unambiguous slice?
- What gets left out when you only train on the easy cases?
- How would you design the labelling process to capture the ambiguous cases without introducing noise?

---

🎓 **Complete** — your coach will direct you next. Your coach will direct you on whether to do the **Going Further** extensions or move straight to **Activity 7 (Reflection)** — it depends on time and energy in the room. Extensions tend to work better before the reflective wind-down, but either order is fine.
