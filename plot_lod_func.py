# %%
import matplotlib.pyplot as plt
from src.sensitivity_sampling import SensitivitySampling


# %%
def gen_npoints(num_lods):
    ss = SensitivitySampling(None)
    npoints = []

    for i in range(num_lods):
        n = ss.get_n_points_for_lod(num_lods, i, 1_000_000)
        npoints.append(n)

    return npoints


# %%
def plot_func(data, logarithmic=False):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    scale = "linear"

    if logarithmic:
        scale = "logarithmic"
        plt.yscale("log")

    ax.plot(range(num_lods), npoints, marker="o", linestyle="-", color="b")

    fig.suptitle(f"Number of Points per LOD (k_max = 1000000), {scale} scale")
    fig.show()


# %%
num_lods = 10
npoints = gen_npoints(num_lods)

# %%
plot_func(npoints)
plot_func(npoints, logarithmic=True)

# %%
