import os
import math
import matplotlib.pyplot as plt
import numpy as np

# User parameters
n_points = 14174243  # Set to match your data or change as needed
num_lods = 5
lower_bound_compression = 0.005
upper_bound_compression = 0.7

# Directory containing benchmark data
BENCHMARK_DIR = '.'

# Dynamically find all subfolders with a num_points file
FUNC_NAMES = []
COLORS = []
for i, folder in enumerate(sorted(os.listdir(BENCHMARK_DIR))):
    folder_path = os.path.join(BENCHMARK_DIR, folder)
    if os.path.isdir(folder_path) and os.path.exists(os.path.join(folder_path, 'num_points')):
        FUNC_NAMES.append(folder)
        COLORS.append(f'tab:{["blue", "orange", "green", "red", "purple", "brown", "pink", "gray", "olive", "cyan"][i % 10]}')

# LOD calculation functions
def get_n_points_for_lod(num_lods, lod_level, n_points, func, lower_bound_compression, upper_bound_compression):
    min_point_count = n_points * lower_bound_compression
    max_point_count = n_points * upper_bound_compression
    if func == 'linear':
        if num_lods == 1:
            return int(max_point_count)
        step = (max_point_count - min_point_count) / (num_lods - 1)
        n_lod_points = min_point_count + step * lod_level
        return int(round(n_lod_points))
    elif func == 'logarithmic':
        if num_lods == 1:
            return int(max_point_count)
        log_min = math.log(1)
        log_max = math.log(num_lods)
        log_lod = math.log(lod_level + 1)
        n_lod_points = min_point_count + (max_point_count - min_point_count) * (log_lod - log_min) / (log_max - log_min)
        return int(round(n_lod_points))
    elif func == 'exponential':
        if num_lods == 1:
            return int(max_point_count)
        exp_min = 1
        exp_max = math.exp(num_lods - 1)
        exp_lod = math.exp(lod_level)
        n_lod_points = min_point_count + (max_point_count - min_point_count) * (exp_lod - exp_min) / (exp_max - exp_min)
        return int(round(n_lod_points))
    elif func == 'exponential2':
        if num_lods == 1:
            return int(max_point_count)
        exp_divisor = 0.5  # Steep version
        exp_min = 1
        exp_max = math.exp((num_lods - 1) / exp_divisor)
        exp_lod = math.exp(lod_level / exp_divisor)
        n_lod_points = min_point_count + (max_point_count - min_point_count) * (exp_lod - exp_min) / (exp_max - exp_min)
        return int(round(n_lod_points))
    else:
        raise ValueError(f"Invalid function: {func}")

def read_benchmark_points(func_name):
    path = os.path.join(BENCHMARK_DIR, func_name, 'num_points')
    if not os.path.exists(path):
        return None
    lods = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('Generating LOD'):
                parts = line.split()
                lod_level = int(parts[2])
                n_points = int(parts[-2])
                lods.append((lod_level, n_points))
    return lods

def main():
    plt.figure(figsize=(10, 6))
    lod_levels = np.arange(num_lods)
    smooth_lod_levels = np.linspace(0, num_lods - 1, 100)
    handles = []
    labels = []
    for i, func in enumerate(FUNC_NAMES):
        color = COLORS[i % len(COLORS)]
        # Theoretical smooth curve (as percentage)
        curve = [get_n_points_for_lod(num_lods, lod, n_points, func, lower_bound_compression, upper_bound_compression) / n_points * 100 for lod in smooth_lod_levels]
        line, = plt.plot(smooth_lod_levels, curve, color=color, label=f"{func}")
        # Benchmark data (dots, not in legend, as percentage)
        bench = read_benchmark_points(func)
        if bench:
            bench_lods, bench_points = zip(*bench)
            bench_points_percent = [p / n_points * 100 for p in bench_points]
            plt.scatter(bench_lods, bench_points_percent, color=color, marker='o', s=60, zorder=3)
        handles.append(line)
        labels.append(f"{func}")
    plt.xlabel('LOD Level')
    plt.ylabel('Percentage of Points (%)')
    plt.title('LOD Point Calculation Curves')
    plt.xticks(lod_levels)  # Only show 0,1,2,3,4
    plt.ylim(lower_bound_compression * 100, upper_bound_compression * 100)
    plt.grid(True)
    plt.legend(handles, labels, title='Function')
    plt.tight_layout()
    plt.savefig('lod_curves.png')
    plt.show()

if __name__ == '__main__':
    main()
