import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
from seaborn import heatmap


def read_correlation_coefficients(file_paths):
    correlation_data = {}
    for dataset, file_path in file_paths.items():
        with open(file_path, "r") as file:
            reader = csv.DictReader(file)
            for row in reader:
                if "Sequence" not in row:
                    sequence = "mprage"
                else:
                    sequence = row["Sequence"]
                metric = row["Metric"]
                corr = float(row["Correlation Coefficient"])
                p_val = float(row["P-Value"])
                if sequence not in correlation_data:
                    correlation_data[sequence] = {}
                if metric not in correlation_data[sequence]:
                    correlation_data[sequence][metric] = {}
                if dataset not in correlation_data[sequence][metric]:
                    correlation_data[sequence][metric][dataset] = []
                correlation_data[sequence][metric][dataset].append((corr, p_val))
    return correlation_data


def plot_comparison_heatmaps(correlation_data, save_dir=None):

    # setting the colormap to be grey between -0.6 and 0.6
    coolwarm = plt.get_cmap('coolwarm')
    limit = 0.7
    red_part = coolwarm(np.linspace(limit, 1, 128))
    blue_part = coolwarm(np.linspace(0, 1-limit, 128))
    length = int(128/(1-limit)*limit * 2)
    grey = np.full((length, 4), 0.95)
    colors = np.vstack([blue_part, grey, red_part])
    cmap = mcolors.LinearSegmentedColormap.from_list('cmap', colors)

    sequences = list(correlation_data.keys())
    metrics = list(next(iter(correlation_data.values())).keys())

    heatmap_data = np.full((len(sequences)+1, len(metrics)), np.nan)
    mask = np.ones_like(heatmap_data, dtype=bool)
    for j, metric in enumerate(metrics):
        corr_values = correlation_data["mprage"][metric]["cubric"]
        mean_corr = np.mean(
            [corr for corr, p_val in corr_values if p_val < 0.05])
        if not np.isnan(mean_corr):
            heatmap_data[0, j] = mean_corr
            mask[0, j] = False

    for i, sequence in enumerate(sequences):
        for j, metric in enumerate(metrics):
            corr_values = correlation_data[sequence][metric]["nru"]
            mean_corr = np.mean([corr for corr, p_val in corr_values if p_val < 0.05])
            if not np.isnan(mean_corr):
                heatmap_data[i+1, j] = mean_corr
                mask[i+1, j] = False

    sequ_dict = {"t1tirm": "TIRM", "flair": "FLAIR", "t2tse": "TSE"}
    sequences = ["MPRAGE\n(CUBRIC)", "MPRAGE\n(NRU)"] + [sequ_dict[s] for s in sequences[1:]]

    fig_width = len(metrics) * 1.5 + 1.5
    fig_height = len(sequences) * 1.5
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    heatmap = sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap=cmap,
                          xticklabels=metrics,
                          yticklabels=sequences,
                          ax=ax, square=True,
                          cbar_kws={"shrink": 0.8}, vmin=-1, vmax=1,
                          linewidths = 0.5, linecolor = 'lightgray'
                          )

    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=22)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=22)
    for text in heatmap.texts:
        text.set_size(22)
    cbar = heatmap.collections[0].colorbar
    cbar.ax.tick_params(labelsize=20)
    cbar.set_ticks([-1, -0.5, 0, 0.5, 1])
    cbar.set_label('$\\rho$', fontsize=22, rotation=0)
    plt.subplots_adjust(left=0.11, right=0.99, top=0.9, bottom=0.1)
    if save_dir is not None:
        os.makedirs(os.path.dirname(save_dir), exist_ok=True)
        plt.savefig(f"{save_dir}Heatmap.png")
    plt.show()


def main():
    file_paths = {
        "nru": "./Results-ISMRM25/correlation_scores/openneuro_baseline.csv",
        "cubric": "./Results-ISMRM25/correlation_scores/cubric_baseline.csv",
    }

    save_dir = f"./Results/Compare_Datasets/"
    correlation_data = read_correlation_coefficients(file_paths)
    plot_comparison_heatmaps(correlation_data, save_dir=save_dir)
    print("Done!")

if __name__ == "__main__":
    main()