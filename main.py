# %%
from src.point_cloud import PointCloud
from src.sensitivity_sampling import SensitivitySampling
from src.visualization.app import App
from matplotlib import pyplot as plt
import numpy as np
import os

# %%
print(os.getcwd())
pc = PointCloud.from_laz_file("data/ljubljanski_grad.las")
# pc = PointCloud.from_laz_file("data/GK_462_100.laz")
print(f"{len(pc.points_x)} points")

# %%
ss = SensitivitySampling(pc)

# %%
coreset, weights = ss.sample(1000)

# %%
print(f"coreset: {coreset}")
print(f"weights: {weights}")

# %%
coreset_x = [p[0] for p in coreset]
coreset_y = [p[1] for p in coreset]
coreset_z = [p[2] for p in coreset]

# %%
# plot
fig = plt.figure()
ax = fig.add_subplot(projection="3d")

ax.scatter(coreset_x, coreset_y, coreset_z, cmap="viridis")
fig.show()


# %%
print(pc.get_labels())
print(np.unique(pc.get_labels(), return_counts=True))
print(pc.get_cluster_centers())
print(pc.get_squared_distances())

# %%
app = App(pc)
app.run()

# %%
