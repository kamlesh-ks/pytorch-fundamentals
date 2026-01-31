# PyTorch Beginner Learning Plan

## Overview
This is a structured 8-module learning plan designed for beginners. Each module builds on the previous one, with hands-on code examples you can run and modify.

## Prerequisites
- Basic Python knowledge (variables, functions, loops, classes)
- Basic math (algebra, matrices basics)
- Python 3.8+ installed

## Setup Instructions

### Install PyTorch
```bash
# For CPU only (simpler for beginners)
pip install torch torchvision

# For GPU support (if you have NVIDIA GPU)
# Visit https://pytorch.org/get-started/locally/ for the right command
```

### Verify Installation
```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
```

---

## Learning Modules

### Module 1: Tensors - The Foundation (Start Here!)
Folder: `01_tensors/`

**What you'll learn:**
- What tensors are (think of them as powerful arrays)
- Creating tensors
- Tensor attributes (shape, dtype, device)
- Basic operations

**Key concepts:**
- Tensors are like NumPy arrays but can run on GPU
- They are the basic building blocks of all PyTorch programs

---