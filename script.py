import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scanpy as sc

from pyCHIRAL import CHIRAL, ccg, optimal_shift

# path = "/Users/salati/Documents/CODE/github/scCircadianMeta/data/BULK/adata_liver_bulk.h5ad"


path = "/Users/salati/Documents/CODE/github/scCircadianMeta/data/BULK/zhang.h5ad"
adata = sc.read_h5ad(path)
organ = "liver"
cc = np.median(adata.X.sum(axis=1))
eps = 1 / cc
adata.layers["s_log"] = np.log(adata.X + eps)
adata2 = adata[adata.obs.organ == organ]
ccg = np.intersect1d(adata2.var_names, ccg)

res = CHIRAL(adata2, clockgenes=ccg, layer="s_log")
true_phase = (adata2.obs.ZT % 24) * 2 * np.pi / 24

phi = res["phi"]
plt.scatter(true_phase, phi)
