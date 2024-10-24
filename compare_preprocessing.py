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
                metric = row["Metric"]
                corr = float(row["Correlation Coefficient"])
                p_val = float(row["P-Value"])
                if metric not in correlation_data:
                    correlation_data[metric] = {}
                if setting not in correlation_data[metric]:
                    correlation_data[metric][setting] = []
                correlation_data[metric][setting].append((corr, p_val))
    return correlation_data

def plot_comparison_heatmaps(correlation_data, out_dirs):
    # setting the colormap to be grey between -0.6 and 0.6
    coolwarm = plt.get_cmap('coolwarm')
    red_part = coolwarm(np.linspace(0.6, 1, 128))
    blue_part = coolwarm(np.linspace(0, 0.4, 128))
    grey = np.full((384, 4), 0.95)
    colors = np.vstack([blue_part, grey, red_part])
    cmap = mcolors.LinearSegmentedColormap.from_list('cmap', colors)

    metrics = list(correlation_data.keys())
    settings = list(out_dirs.keys())


    heatmap_data = np.full((len(settings), len(metrics)), np.nan)
    mask = np.ones_like(heatmap_data, dtype=bool)
    for i, setting in enumerate(settings):
        for j, metric in enumerate(metrics):
            print(correlation_data[metric].keys())
            corr_values = correlation_data[metric][setting]
            mean_corr = np.mean([corr for corr, p_val in corr_values if p_val < 0.05])
            if not np.isnan(mean_corr):
                heatmap_data[i, j] = mean_corr
                mask[i, j] = False

    fig, ax = plt.subplots(figsize=(len(metrics) * 1.5, len(settings) * 1.5))
    heatmap = sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap=cmap,
                            xticklabels=metrics, yticklabels=settings, ax=ax,
                            square=True, cbar_kws={"shrink": 0.9}, vmin=-1,
                            vmax=1, mask=mask)
    # ax.set_xlabel('Metrics', fontsize=18)
    # ax.set_ylabel('Preprocessing Settings', fontsize=18)
    # ax.set_title(f'Comparison of Correlation Coefficients for CUBRIC data',
    #                 fontsize=18)
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
    
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors

def plot_comparison_heatmaps_v2(correlation_data, out_dirs):
    # Setting the colormap to be grey between -0.6 and 0.6
    coolwarm = plt.get_cmap('coolwarm')
    red_part = coolwarm(np.linspace(0.6, 1, 128))
    blue_part = coolwarm(np.linspace(0, 0.4, 128))
    grey = np.full((384, 4), 0.95)
    colors = np.vstack([blue_part, grey, red_part])
    cmap = mcolors.LinearSegmentedColormap.from_list('cmap', colors)

    metrics = list(correlation_data.keys())
    settings = list(out_dirs.keys())

    heatmap_data = np.full((len(metrics), len(settings)), np.nan)
    mask = np.ones_like(heatmap_data, dtype=bool)

    for i, metric in enumerate(metrics):
        for j, setting in enumerate(settings):
            corr_values = correlation_data[metric][setting]
            mean_corr = np.mean([corr for corr, p_val in corr_values if p_val < 0.05])
            if not np.isnan(mean_corr):
                heatmap_data[i, j] = mean_corr
                mask[i, j] = False

    # Without yticks for paper
    fig, ax = plt.subplots(figsize=(len(settings) * 2, len(metrics) * 1.2))  # Reduced height
    heatmap = sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap=cmap,
                            xticklabels=settings, yticklabels=False, ax=ax,
                            square=True, cbar=False, vmin=-1, vmax=1, mask=mask,
                            linewidths=0.5, linecolor='lightgray')  # Light gray borders

    ax.set_xticklabels(ax.get_xticklabels(), fontsize=14, rotation=20, ha='center')
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')
    # ax.set_yticklabels(ax.get_yticklabels(), fontsize=14, rotation=0)

    # Increase the size of text inside the boxes
    for text in heatmap.texts:
        text.set_size(14)  # Increased text size

    # Add colorbar closer to the heatmap
    cbar = fig.colorbar(heatmap.collections[0], ax=ax, orientation='horizontal', pad=0.02)
    cbar.ax.tick_params(labelsize=12)
    cbar.set_ticks([-1, -0.5, 0, 0.5, 1])

    plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.2)  # Adjust margins
    plt.show()

    # With ticks
    fig, ax = plt.subplots(figsize=(len(settings) * 2, len(metrics) * 1.2))  # Reduced height
    heatmap = sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap=cmap,
                            xticklabels=settings, yticklabels=metrics, ax=ax,
                            square=True, cbar=False, vmin=-1, vmax=1, mask=mask,
                            linewidths=0.5, linecolor='lightgray')  # Light gray borders

    ax.set_xticklabels(ax.get_xticklabels(), fontsize=14, rotation=20, ha='center')
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=14, rotation=0)

    # Increase the size of text inside the boxes
    for text in heatmap.texts:
        text.set_size(14)  # Increased text size

    # Add colorbar closer to the heatmap
    cbar = fig.colorbar(heatmap.collections[0], ax=ax, orientation='horizontal', pad=0.02)
    cbar.ax.tick_params(labelsize=12)
    cbar.set_ticks([-1, -0.5, 0, 0.5, 1])

    plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.2)  # Adjust margins
    plt.show()


def main():
    out_dirs = {
        # Reduction:
        "Worst": "./Results/CUBRICdata/2024-10-21_22-14/", #baseline
        "Mean": "./Results/CUBRICdata/2024-10-22_00-49/",
        # # # Brain mask:
        # "Mask": "./Results/CUBRICdata/2024-10-21_22-14/",  #baseline
        # "Multiply": "./Results/CUBRICdata/2024-10-22_03-24/",  ####
        # "No Mask": "./Results/CUBRICdata/2024-10-22_12-53/",  ###
        # # # # Normalisation
        # "Percentile": "./Results/CUBRICdata/2024-10-21_22-14/",  # baseline
        # "Min-Max": "./Results/CUBRICdata/2024-10-22_19-18/",
        # "Mean-Std": "./Results/CUBRICdata/2024-10-22_21-58/",
        # "None": "./Results/CUBRICdata/2024-10-22_22-14/",
    }

    correlation_data = read_correlation_coefficients(out_dirs)
    plot_comparison_heatmaps_v2(correlation_data, out_dirs)

    print("Done!")

if __name__ == "__main__":
    main()