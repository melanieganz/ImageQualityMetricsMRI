import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
        sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="coolwarm",
                    xticklabels=metrics, yticklabels=settings, ax=ax, square=True,
                    cbar_kws={"shrink": 0.5}, vmin=-1, vmax=1, mask=mask)
        ax.set_xlabel('Metrics', fontsize=14)
        ax.set_ylabel('Preprocessing Settings', fontsize=14)
        ax.set_title(f'Comparison of Correlation Coefficients for {sequence}',
                     fontsize=16)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=12)
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=12)
        plt.show()

def main():
    out_dirs = {
        # "worst": "./Results/OpenNeuro/2024-08-28_08-19/",
        # "mean": "./Results/OpenNeuro/2024-08-22_08-55/",
        "None": "./Results/OpenNeuro/2024-08-27_16-52/",
        "Mask": "./Results/OpenNeuro/2024-08-28_08-19/",
        "Multiply": "./Results/OpenNeuro/2024-08-27_15-21/",
        # "None": "./Results/OpenNeuro/2024-08-28_07-33/",
        # "min_max": "./Results/OpenNeuro/2024-08-28_08-19/",
        # "mean_std": "./Results/OpenNeuro/2024-08-28_07-31/",
        # "percentile": "./Results/OpenNeuro/2024-09-19_11-57/",
    }

    correlation_data = read_correlation_coefficients(out_dirs)
    plot_comparison_heatmaps(correlation_data, out_dirs)

    print("Done!")

if __name__ == "__main__":
    main()