import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import scanpy as sc

from pyCHIRAL.chiral import CHIRAL

path = "/Users/salati/Documents/CODE/github/scCircadianMeta/data/BULK/adata_liver_bulk.h5ad"


path = "/Users/salati/Documents/CODE/github/scCircadianMeta/data/BULK/zhang.h5ad"
adata = sc.read_h5ad(path)

organ = "liver"


cc = np.median(adata.X.sum(axis=1))
eps = 1 / cc
adata.layers["s_log"] = np.log(adata.X + eps)

adata2 = adata[adata.obs.organ == organ]

res = CHIRAL(adata2, layer="s_log", iterations=500)

true_phase = (adata2.obs.ZT % 24) * 2 * np.pi / 24

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
