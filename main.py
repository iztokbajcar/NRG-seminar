# %%
from src.point_cloud import PointCloud
from src.sensitivity_sampling import SensitivitySampling
import os

# %%
print(os.getcwd())
pc = PointCloud.from_laz_file("data/test.laz")

# %%
ss = SensitivitySampling(pc.to_array())

# %%
results = ss.sample()

# %%
print(results.labels_)

# %%
