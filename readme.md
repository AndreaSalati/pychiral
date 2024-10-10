# pyCHIRAL

**pyCHIRAL** is a Python implementation of **CHIRAL** (Circular HIerarchical Reconstruction ALgorithm), an R package originally developed by Lorenzo Talamanca, designed to infer circadian clock phases from RNA-seq data. This algorithm is based on **Expectation Maximization** and utilizes a **Statistical Physics-inspired approach**.

## Features

- Efficiently infers circadian clock phases from RNA-seq data.
- Incorporates an advanced Expectation Maximization algorithm.
- Inspired by methods from Statistical Physics.
  
This method was applied in the following Science paper: [Circadian Phases Inference Using RNA-seq](https://www.science.org/doi/10.1126/science.add0846). Further information regarding the alorithm can be found in the supplement.

For the original R version of this package, visit the **CHIRAL** repository [here](https://github.com/naef-lab/CHIRAL/tree/master/Pkg/CHIRAL).

## Installation

To run pyCHIRAL, you need to install the following dependencies:

```bash
pip install tqdm numpy pandas scipy