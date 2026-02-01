# Module 3: Autograd - Automatic Differentiation

## What is Autograd?

**Autograd** is PyTorch's automatic differentiation engine. It automatically computes gradients (derivatives) for you - this is the magic that makes neural networks learn!

## Why Do We Need Gradients?

Neural networks learn by:
1. Making predictions
2. Measuring how wrong they are (loss)
3. Figuring out how to adjust weights to reduce error (gradients)
4. Updating weights in the right direction

Gradients tell us: "If I change this weight a little, how much will the error change?"

## The Key Concept

```
y = f(x, w)  # Some function with weights w
loss = error(y, target)  # How wrong we are

# Autograd computes: d(loss)/d(w)
# "How does loss change when w changes?"
```

## Files in This Module

1. `01_gradients_basics.py` - What gradients are
2. `02_computational_graph.py` - How PyTorch tracks operations
3. `03_practical_example.py` - Using gradients in practice

## This is THE Core of Deep Learning!

Understanding autograd is understanding how neural networks learn.
