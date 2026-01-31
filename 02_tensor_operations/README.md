# Module 2: Advanced Tensor Operations

## What You'll Learn

Building on Module 1, you'll master:
- Indexing and slicing (accessing specific elements)
- Reshaping tensors (changing dimensions)
- Combining tensors (stacking, concatenating)
- Broadcasting (operations on different-shaped tensors)

## Why This Matters

Real neural networks constantly reshape and combine data:
- Images need reshaping for different layers
- Batches of data are stacked together
- Outputs are combined for final predictions

## Files in This Module

1. `01_indexing_slicing.py` - Access specific elements
2. `02_reshaping.py` - Change tensor dimensions
3. `03_combining_tensors.py` - Stack and concatenate
4. `exercises.py` - Practice problems

## Key Concept: Shape is Important!

Most PyTorch errors come from shape mismatches. Master these operations to debug efficiently.

```python
# Always check shapes when confused!
print(f"Tensor shape: {tensor.shape}")
```
