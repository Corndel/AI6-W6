# Fine‑Tuning in Plain English (AI6 Workshop 6W)

If you only remember one thing:

**Fine‑tuning is not rebuilding a model.** It is continuing optimisation from a strong starting point.

---

## The story

Imagine you hire a graduate who already speaks excellent English. They understand grammar, tone, and context because they’ve read a huge amount of text.

A pretrained model is like that graduate: it already has strong general ability.

Now you hire them into a finance team. They still speak English, but they don’t yet have good instincts for phrases like “guidance cut”, “beat expectations”, or “downgraded outlook”.

**Fine‑tuning** is onboarding them into your domain. You are not teaching them English again — you are sharpening their behaviour for your environment.

---

## What happens under the hood (plain English, with the correct maths words)

When you run `trainer.train()` the model repeats a loop.

It reads a batch of labelled examples, makes predictions, and gets scored by a **loss function** (a differentiable “how wrong was I?” number).

Because the loss is differentiable, we can compute its **gradient** — the slope (a *partial derivative*) that tells us which tiny changes to the weights would reduce loss.

An optimiser (often **AdamW** for transformers) uses those gradients to nudge the model’s weights by a small amount. This is in the family of **stochastic gradient descent (SGD)**: we estimate the gradient using mini‑batches rather than the full dataset.

Then we repeat this thousands of times.

That’s optimisation. The model isn’t “thinking” — it’s adjusting numbers to reduce loss.

---

## Fine‑tuning vs retraining vs optimising

**Optimising** is the maths engine: reduce a loss function by adjusting weights using gradients.

**Fine‑tuning** means using that engine starting from a pretrained model (good starting point), using your domain data.

**Retraining** usually means running optimisation again because your data changed (new data, drift, new labels). Sometimes people use “retraining” to mean training from scratch, but that is rarer in business because it is expensive.

A helpful metaphor:
- Fine‑tuning is tailoring a suit that already fits.
- Training from scratch is making a new suit from fabric.

---

## Does fine‑tuning erase previous knowledge?

Usually, **no**. Fine‑tuning does not wipe the model’s “general English” knowledge — it makes incremental updates.

However, models can become overly specialised if training is aggressive (very high learning rate, too many epochs, very narrow data). This is sometimes called **catastrophic forgetting**.

In engineering terms: you are moving the model around in weight‑space. Fine‑tuning is a small move. Aggressive training is a big move.

---

## What should I look at while training?

Think of training as a dashboard with two key signals:

**Loss is altitude.** If training loss goes down, optimisation is working mathematically.

**Validation performance is the road test.** Accuracy/F1 tell you whether the model behaves better on examples it hasn’t seen.

The key relationship is the **generalisation gap**:

> Generalisation gap = the difference between training performance and validation performance.  
> If training improves but validation does not, the model is learning the training set too specifically.

---

## When should I stop tweaking?

A simple, practical rule:

1) get a baseline  
2) change **one dial**  
3) measure  
4) log the result  
5) stop when improvements are small relative to time/compute cost

This is not just “maths” — it’s professional engineering discipline.
