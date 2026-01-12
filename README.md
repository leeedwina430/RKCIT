# RKCIT
This repository contains the official implementation of **RHSIC** and **RKCIT** from the paper:
> **Extracting Rare Dependence Patterns via Adaptive Sample Reweighting** (ICML 2025).

## Environment Setup
The code requires Python 3.x. You can install the necessary dependencies using the following commands:

### 1. Install general dependencies
```bash
pip install scipy tqdm torch scikit-learn causal-learn
```

### 2. Install kerpy
This project depends on kerpy. You can install it directly from GitHub:
```bash
pip install git+[https://github.com/oxcsml/kerpy.git](https://github.com/oxcsml/kerpy.git)
```
(Or refer to the official repository: https://github.com/oxcsml/kerpy)

## Quick Start
To run a simple demo of RHSIC:
```bash
python demo.py
```
It includes a usage example as well.
