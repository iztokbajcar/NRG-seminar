# %%
from src.visualization.app import App
from src.sensitivity_sampling import SensitivitySampling
from src.point_cloud import PointCloud

# %%
maps = ["ljubljana.laz", "ljubljanski_grad.las"]


# %%
def benchmark(map_name, n_lods, tiles_dim):
    pc = PointCloud.from_laz_file(f"data/{map_name}")

    # generate LODs
    ss = SensitivitySampling(pc)
    ss.generate_lods(n_lods)

    # generate tiles
    tiles_dir = f"data/{map_name}_tiles"
    pc.save_as_tiles(tiles_dim[0], tiles_dim[1], tiles_dir)

    # run app
    app = App(tiles_dir, tiles_dim, n_lods, benchmark=True)
    app.run()

    return app.benchmark_results


# %%
results = benchmark("ljubljanski_grad.las", 5, (10, 10))

# %%
print(results)

# %%
