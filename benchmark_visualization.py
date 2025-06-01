# %%
from src.visualization.app import App
from src.sensitivity_sampling import SensitivitySampling
from src.point_cloud import PointCloud
import json
import matplotlib.pyplot as plt
import numpy as np
import os


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
# plot results
def plot_results(results, map_name, tiles_dim):
    if not os.path.exists("plots"):
        os.makedirs("plots")

    if not os.path.exists(os.path.join("measurements", "visualization")):
        os.makedirs(os.path.join("measurements", "visualization"))

    fig = plt.figure()
    ax = fig.add_subplot(111)

    lods_load = list(results["loading"].keys())
    times_load = list(results["loading"].values())
    times_load = list(map(lambda l: np.mean(l), times_load))

    lods_gpu = list(results["gpu_upload"].keys())
    times_gpu = list(results["gpu_upload"].values())
    times_gpu = list(map(lambda l: np.mean(l), times_gpu))

    ax.plot(lods_load, times_load, marker="o", linestyle="-", color="b")
    ax.plot(lods_gpu, times_gpu, marker="o", linestyle="-", color="r")
    fig.suptitle(
        f"Average tile loading times for map {map_name}, {num_lods} LODs, grid size {tiles_dim[0]}x{tiles_dim[1]}"
    )
    ax.set_xlabel("LOD Level")
    ax.set_ylabel("Time (seconds)")
    ax.legend(["Load from disk", "GPU Upload"])
    plt.show()

    fig.savefig(
        f"plots/{map_name}_tiles_{tiles_dim[0]}x{tiles_dim[1]}_lods_{num_lods}.png"
    )

    with open(
        f"measurements/visualization/{map_name}_tiles_{tiles_dim[0]}x{tiles_dim[1]}_lods_{num_lods}.json",
        "w",
    ) as f:
        json.dump(results, f)


# %%
for map_name in ["ljubljanski_grad.las", "ljubljana.laz"]:
    for tiles_dim in [(5, 5), (10, 10), (1, 5), (1, 10)]:
        for num_lods in [5, 10]:
            print(
                f"Benchmarking {map_name} with tile dimensions {tiles_dim} and {num_lods} LODs"
            )
            results = benchmark(map_name, num_lods, tiles_dim)
            plot_results(results, map_name, tiles_dim)


# %%
