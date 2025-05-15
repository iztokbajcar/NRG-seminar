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
print(pc.labels)
print(np.unique(pc.labels, return_counts=True))
print(pc.cluster_centers)

# %%
