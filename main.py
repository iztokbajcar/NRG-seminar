# %%
from src.point_cloud import PointCloud
from src.sensitivity_sampling import SensitivitySampling
import numpy as np
import os

# %%
print(os.getcwd())
pc = PointCloud.from_laz_file("data/test.laz")
print(f"{len(pc.points_x)} points")

# %%
ss = SensitivitySampling(pc)

# %%
ss.sample()

# %%
print(pc.get_labels())
print(np.unique(pc.get_labels(), return_counts=True))
print(pc.get_cluster_centers())
print(pc.get_kmeans_cost())

# %%
