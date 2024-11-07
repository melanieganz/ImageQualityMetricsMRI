import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors


def read_correlation_coefficients(out_dirs):
    correlation_data = {}
    for setting, out_dir in out_dirs.items():
        if ".csv" not in out_dir:
            file_path = os.path.join(out_dir, "correlation_coefficients.csv")
        else:
            file_path = out_dir
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
                if setting not in correlation_data[sequence][metric]:
                    correlation_data[sequence][metric][setting] = []
                correlation_data[sequence][metric][setting].append((corr, p_val))
    return correlation_data

def plot_comparison_heatmaps(correlation_data, out_dirs, show_yticklabels=True,
                             save_dir=None):
    # setting the colormap to be grey between -limit and limit
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
    settings = list(out_dirs.keys())

    for sequence in sequences:
        heatmap_data = np.full((len(metrics), len(settings)), np.nan)
        mask = np.ones_like(heatmap_data, dtype=bool)
        for i, metric in enumerate(metrics):
            for j, setting in enumerate(settings):
                if metric in ["FSIM", "VIF", "LPIPS"]:
                    if setting == "Mask":
                        continue
                corr_values = correlation_data[sequence][metric][setting]
                mean_corr = np.mean([corr for corr, p_val in corr_values if p_val < 0.05])
                if not np.isnan(mean_corr):
                    heatmap_data[i, j] = mean_corr
                    mask[i, j] = False

        fig, ax = plt.subplots(figsize=(len(settings) * 2, len(metrics) * 1.2))
        ytick = metrics if show_yticklabels else False
        heatmap = sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap=cmap,
                    xticklabels=settings, yticklabels=ytick, ax=ax, square=True,
                    cbar=False, vmin=-1, vmax=1, mask=mask,
                    linewidths = 0.5, linecolor = 'lightgray')
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=18, rotation=20,
                           ha='center')
        ax.xaxis.set_ticks_position('top')
        ax.xaxis.set_label_position('top')
        if show_yticklabels:
            ax.set_yticklabels(ax.get_yticklabels(), fontsize=18, rotation=0)

        for text in heatmap.texts:
            text.set_size(18)

        plt.subplots_adjust(left=0.01, right=1.45, top=0.9, bottom=0.1)
        if save_dir is not None:
            os.makedirs(os.path.dirname(save_dir), exist_ok=True)
            plt.savefig(f"{save_dir}_{sequence}.png")
        plt.show()

        # # create colorbar
        # fig, ax = plt.subplots(figsize=(12, 6))
        # plt.axis('off')
        # cbar = fig.colorbar(heatmap.collections[0], ax=ax,
        #                     orientation='horizontal',
        #                     pad=0.02)
        # cbar.ax.tick_params(labelsize=22)
        # cbar.set_ticks([-1, -0.5, 0, 0.5, 1])
        # cbar.set_label('$\\rho$', fontsize=22, rotation=0)
        # plt.show()

def main():
    preproc_settings = {
        "reduction":{"Worst": "./results-ismrm25/correlation_scores/"
                              "openneuro_mask_mask.csv",
                     "Mean": "./results-ismrm25/correlation_scores/"
                             "openneuro_reduction_mean.csv"},
        "brain_mask": {"Mask": "./results-ismrm25/correlation_scores/"
                               "openneuro_mask_mask.csv",
                       "Multiply": "./Rresults-ismrm25/correlation_scores/"
                                   "openneuro_baseline.csv",
                       "No Mask": "./results-ismrm25/correlation_scores/"
                                  "openneuro_mask_none.csv"},
        "normalisation": {"Percentile": "./results-ismrm25/correlation_scores/"
                                        "openneuro_mask_mask.csv",
                          "Min-Max": "./results-ismrm25/correlation_scores/"
                                     "openneuro_norm_minmax.csv",
                          "Mean-Std": "./results-ismrm25/correlation_scores/"
                                      "openneuro_norm_meanstd.csv",
                          "None": "./results-ismrm25/correlation_scores/"
                                  "openneuro_norm_none.csv"}
    }
    for settings, out_dirs in preproc_settings.items():

        if "Mask" in out_dirs.keys():
            show_yticklabels = True
        else:
            show_yticklabels = False

        save_dir = f"./Results/Compare_Preproc_OpenNeuro/{settings}"
        correlation_data = read_correlation_coefficients(out_dirs)
        plot_comparison_heatmaps(correlation_data, out_dirs,
                                 show_yticklabels=show_yticklabels,
                                 save_dir=save_dir)

    print("Done!")

    preproc_settings = {
        "reduction":{"Worst": "./results-ismrm25/correlation_scores/"
                              "cubric_mask_mask.csv",
                     "Mean": "./Rresults-ismrm25/correlation_scores/"
                             "cubric_reduction_mean.csv"},
        "brain_mask": {"Mask": "./results-ismrm25/correlation_scores/"
                               "cubric_mask_mask.csv",
                       "Multiply": "./results-ismrm25/correlation_scores/"
                                   "cubric_baseline.csv",
                       "No Mask": "./results-ismrm25/correlation_scores/"
                                  "cubric_mask_none.csv"},
        "normalisation": {"Percentile": "./results-ismrm25/correlation_scores/"
                                        "cubric_mask_mask.csv",
                          "Min-Max": "./results-ismrm25/correlation_scores/"
                                     "cubric_norm_minmax.csv",
                          "Mean-Std": "./results-ismrm25/correlation_scores/"
                                      "cubric_norm_meanstd.csv",
                          "None": "./results-ismrm25/correlation_scores/"
                                  "cubric_norm_none.csv"}
    }
    for settings, out_dirs in preproc_settings.items():

        if "Mask" in out_dirs.keys():
            show_yticklabels = True
        else:
            show_yticklabels = False

        save_dir = f"./Results/Compare_Preproc_Cubric/{settings}"
        correlation_data = read_correlation_coefficients(out_dirs)
        plot_comparison_heatmaps(correlation_data, out_dirs,
                                 show_yticklabels=show_yticklabels,
                                 save_dir=save_dir)


if __name__ == "__main__":
    main()