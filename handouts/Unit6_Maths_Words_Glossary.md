# Unit 6 “Maths Words” Glossary (Minimum Viable Vocabulary)

Unit 6 is the maths unit — but you don’t need to be scared of it.

This page gives you the *minimum* vocabulary you need to read ML tooling, docs, and research papers.

---

## Loss (loss function)

**Loss** is a number that measures how wrong a model is.

Metaphor: **altitude**. If loss goes down, we’re going “downhill” (getting less wrong).

A **loss function** is just the rule for calculating that number.

---

## Differentiable

A function is **differentiable** if it’s smooth enough that we can compute its slope.

Metaphor: a smooth hill where you can feel which way is down (instead of a staircase).

---

## Gradient (partial derivatives)

The **gradient** tells you the slope of the loss function.

More precisely, it’s a collection of **partial derivatives**: one “how does loss change if I tweak this weight?” value for every weight.

Metaphor: standing on a hillside with a compass that points “steepest uphill”. If you walk the opposite way, you go downhill fastest.

---

## Stochastic Gradient Descent (SGD)

**Gradient descent** means “take a step downhill using the gradient”.

**Stochastic** means we estimate the gradient from a small batch of data, so it’s a noisy approximation.

Metaphor: instead of surveying the whole mountain, you take a few local measurements and step in the best direction you can *right now*.

Many modern optimisers (like **AdamW**) are in the “SGD family”: they still use gradients, but with extra tricks.

---

## Learning rate

The **learning rate** is your step size.

Metaphor: tiny steps (slow but stable) vs big leaps (fast but you might overshoot).

---

## Batch / mini‑batch

A **batch** is a small chunk of training examples processed together to compute a gradient update.

Metaphor: learning from a handful of flashcards at a time, not the whole textbook.

---

## Epoch

An **epoch** is one full pass through the training dataset.

---

## Generalisation

**Generalisation** means: does the model work on new, unseen data?

Metaphor: practising past papers vs sitting the real exam.

---

## Generalisation gap

The **generalisation gap** is the difference between training performance and validation performance.

If training improves while validation gets worse, the gap is widening — a classic sign of **overfitting**.

---

## Regularisation / weight decay

Regularisation is “don’t let the model get too wild”.

**Weight decay** is one type of regularisation: it gently discourages very large weights.

Metaphor: **friction** that stops the model from overreacting to noise.
