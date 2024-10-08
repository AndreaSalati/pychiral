import numpy as np
from tqdm import tqdm  # Progress bar
from numpy.linalg import inv, det
import pandas as pd
import matplotlib.pyplot as plt

import scanpy as sc

from chiral import CHIRAL

path = "/Users/salati/Documents/CODE/github/scCircadianMeta/data/BULK/adata_liver_bulk.h5ad"
adata = sc.read_h5ad(path)

res = CHIRAL(adata, layer="s_log", iterations=5)

print("run was successful")

true_phase = adata.obs.ZT * 2 * np.pi / 24

phi = res["phi"]

plt.scatter(true_phase, phi)

# np.apply_along_axis
# create a pd.dataframe that I will import in R
# df = pd.DataFrame(
#     adata.layers["s_log"].T, index=adata.var_names, columns=adata.obs_names
# )

# print(df)
# adata.layers["s_log"]
# df.to_csv("s_log.csv")

# md = adata.obs.ZT
# # convert to float
# md = md.astype(float)
# md.to_csv("ZTmod.csv")
