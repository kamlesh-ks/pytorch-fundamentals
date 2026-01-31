# Module 1: Tensors - The Foundation

## What is a Tensor?

A **tensor** is the fundamental data structure in PyTorch. Think of it as a container for numbers:

| Dimension | Name   | Example                   |
| --------- | ------ | ------------------------- |
| 0D        | Scalar | A single number: `5`      |
| 1D        | Vector | A list: `[1, 2, 3]`       |
| 2D        | Matrix | A table: `[[1,2], [3,4]]` |
| 3D+       | Tensor | Images, videos, etc.      |

## Why Tensors?

1. **GPU Acceleration**: Tensors can run on GPUs for massive speedups
2. **Automatic Differentiation**: PyTorch can compute gradients automatically
3. **Optimized Operations**: Built-in fast mathematical operations

## Files in This Module

1. `01_creating_tensors.py` - How to create tensors
2. `02_tensor_attributes.py` - Shape, dtype, device
3. `03_basic_operations.py` - Math with tensors
4. `exercises.py` - Practice problems

## Key Takeaways

After this module, you should be able to:
- Create tensors from Python lists and NumPy arrays
- Understand tensor shape, dtype, and device
- Perform basic mathematical operations
- Convert between tensors and NumPy arrays

## Run the Examples

```bash
cd 01_tensors
python 01_creating_tensors.py
python 02_tensor_attributes.py
python 03_basic_operations.py
```