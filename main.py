# %%
from src.point_cloud import PointCloud
from src.sensitivity_sampling import SensitivitySampling
from src.visualization.app import App
from matplotlib import pyplot as plt
import numpy as np
import os

# %%
LOD_COUNT = 5

MAP_NAME = "ljubljana.laz"
# MAP_NAME = "ljubljanski_grad.las"

# %%
print(os.getcwd())
pc = PointCloud.from_laz_file(f"data/{MAP_NAME}")
print(f"{len(pc.points_x)} points")

# %%
print(np.unique(pc.get_points_class(), return_counts=True))

# %%
ss = SensitivitySampling(pc)

# %%
# coreset, weights = ss.sample(100)

# # %%
# print(f"coreset: {coreset}")
# print(f"weights: {weights}")

# # %%
# coreset_x = [p[0] for p in coreset]
# coreset_y = [p[1] for p in coreset]
# coreset_z = [p[2] for p in coreset]

# # %%
# # plot
# fig = plt.figure()
# ax = fig.add_subplot(projection="3d")

# ax.scatter(coreset_x, coreset_y, coreset_z, cmap="viridis")
# fig.show()


# # %%
# print(pc.get_cluster_centers())
# print(pc.get_squared_distances())

# %%
ss.generate_lods(LOD_COUNT)
pc.get_lods()

# %%

map_name_no_ext = MAP_NAME.replace('.laz', '').replace('.las', '')
tiles_dir = f"data/{map_name_no_ext}_tiles"

pc.save_as_tiles(10, 10, tiles_dir)

# %%
# tile = PointCloud.from_laz_file("data/ljubljanski_grad_tiles/1_2.laz")

# %%
app = App(tiles_dir, (10, 10), LOD_COUNT)
app.run()

# %%
