# pychiral

**pychiral** is a Python implementation of **CHIRAL** (Circular HIerarchical Reconstruction ALgorithm), an R package originally developed by Lorenzo Talamanca, designed to infer circadian clock phases from RNA-seq data. This algorithm is based on **Expectation Maximization** and utilizes a **Statistical Physics-inspired approach**.

## TO implement:
- posteriors via grid eval
- PCA init

## Features

- Efficiently infers circadian clock phases from RNA-seq data.
- Incorporates Expectation Maximization algorithm.
- Inspired by methods from Statistical Physics.
  
This method was applied in the following Science paper: [Sex-dimorphic and age-dependent organization of 24-hour gene expression rhythms in humans](https://www.science.org/doi/10.1126/science.add0846). Further information regarding the alorithm can be found in the supplement.

For the original R version of this package, visit the **CHIRAL** repository [here](https://github.com/naef-lab/CHIRAL/tree/master/Pkg/CHIRAL).

## Installation

To run pychiral, you need to install the following dependencies:

```bash
pip install .
```

## Usage
pychiral is compatible with anndata, and it receives data in the format Nsamples x Ngenes (unlike the Bulk conventions).
A layer of the dataset needs to be specified by argument. 
The layer passed to CHIRAL needs to be **log-tranformed**. By default the data will also be mean centered and standardized.

