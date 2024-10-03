import numpy as np
from tqdm import tqdm  # Progress bar
from numpy.linalg import inv, det
import pandas as pd

import scanpy as sc

from chiral import CHIRAL

path = "/Users/salati/Documents/CODE/github/scCircadianMeta/data/BULK/adata_liver_bulk.h5ad"
adata = sc.read_h5ad(path)

CHIRAL(adata, layer="s_log")
