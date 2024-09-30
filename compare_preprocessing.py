import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors

def read_correlation_coefficients(out_dirs):
    correlation_data = {}
    for setting, out_dir in out_dirs.items():
        file_path = os.path.join(out_dir, "correlation_coefficients.csv")
        with open(file_path, "r") as file:
            reader = csv.DictReader(file)
            for row in reader:
                sequence = row["Sequence"]
                metric = row["Metric"]
                corr = float(row["Correlation Coefficient"])
                p_val = float(row["P-Value"])
                if sequence not in correlation_data:
                    correlation_data[sequence] = {}
                if metric not in correlation_data[sequence]:
                    correlation_data[sequence][metric] = {}
                if setting not in correlation_data[sequence][metric]:
                    correlation_data[sequence][metric][setting] = []
                correlation_data[sequence][metric][setting].append((corr, p_val))
    return correlation_data

def plot_comparison_heatmaps(correlation_data, out_dirs):
    # setting the colormap to be grey between -0.6 and 0.6
    coolwarm = plt.get_cmap('coolwarm')
    red_part = coolwarm(np.linspace(0.6, 1, 128))
    blue_part = coolwarm(np.linspace(0, 0.4, 128))
    grey = np.full((384, 4), 0.95)
    colors = np.vstack([blue_part, grey, red_part])
    cmap = mcolors.LinearSegmentedColormap.from_list('cmap', colors)

    sequences = list(correlation_data.keys())
    metrics = list(next(iter(correlation_data.values())).keys())
    settings = list(out_dirs.keys())

    for sequence in sequences:
        heatmap_data = np.full((len(settings), len(metrics)), np.nan)
        mask = np.ones_like(heatmap_data, dtype=bool)
        for i, setting in enumerate(settings):
            for j, metric in enumerate(metrics):
                corr_values = correlation_data[sequence][metric][setting]
                mean_corr = np.mean([corr for corr, p_val in corr_values if p_val < 0.05])
                if not np.isnan(mean_corr):
                    heatmap_data[i, j] = mean_corr
                    mask[i, j] = False

        fig, ax = plt.subplots(figsize=(len(metrics) * 1.5, len(settings) * 1.5))
        heatmap = sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap=cmap,
                              xticklabels=metrics, yticklabels=settings, ax=ax,
                              square=True, cbar_kws={"shrink": 0.8}, vmin=-1,
                              vmax=1, mask=mask)
        ax.set_xlabel('Metrics', fontsize=18)
        ax.set_ylabel('Preprocessing Settings', fontsize=18)
        ax.set_title(f'Comparison of Correlation Coefficients for {sequence}',
                     fontsize=18)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=18)
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=18)
        for text in heatmap.texts:
            text.set_size(16)
        cbar = heatmap.collections[0].colorbar
        cbar.ax.tick_params(labelsize=16)
        cbar.set_label('$\\rho$', fontsize=16, rotation=0)
        plt.show()

        # without title etc. for the paper:
        fig, ax = plt.subplots(figsize=(len(metrics) * 1.5, len(settings) * 1.6))
        heatmap = sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap=cmap,
                    xticklabels=metrics, yticklabels=settings, ax=ax, square=True,
                    cbar=False, vmin=-1, vmax=1, mask=mask)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=20)
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=20)

        for text in heatmap.texts:
            text.set_size(18)
        plt.subplots_adjust(left=0.11, right=0.95, top=0.9, bottom=0.1)
        plt.show()

def main():
    out_dirs = {
        # Reduction:
        "Worst": "./Results/OpenNeuro/2024-09-27_12-35/", #baseline
        "Mean": "./Results/OpenNeuro/2024-09-27_16-25/",
        # Brain mask:
        # "Mask": "./Results/OpenNeuro/2024-09-27_12-35/",  #baseline
        # "Multiply": "./Results/OpenNeuro/2024-09-30_09-21/",  ####
        # "None": "./Results/OpenNeuro/2024-09-30_10-39/",  ###
        # Normalisation
        # "Percentile": "./Results/OpenNeuro/2024-09-27_12-35/",  # baseline
        # "Min-Max": "./Results/OpenNeuro/2024-09-27_20-44/",
        # "Mean-Std": "./Results/OpenNeuro/2024-09-27_22-12/",
        # "None": "./Results/OpenNeuro/2024-09-27_23-08/",
    }

    correlation_data = read_correlation_coefficients(out_dirs)
    plot_comparison_heatmaps(correlation_data, out_dirs)

    print("Done!")

if __name__ == "__main__":
    main()