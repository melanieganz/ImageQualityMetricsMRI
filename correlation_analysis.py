import os.path
import csv
import numpy as np
import argparse
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors


def load_metrics_scores(input_csv):
    full_reference_metrics = ["SSIM", "PSNR", "FSIM", "VIF", "LPIPS"]
    reference_free_metrics = ["AES", "TG", "NGS", "GE", "IE"]
    observer_scores_to_include = ["Score"]
    

    metrics = {"full_ref": {m: [] for m in full_reference_metrics},
               "ref_free": {m: [] for m in reference_free_metrics}}
    observer_scores = {"full_ref": {o: [] for o in observer_scores_to_include},
                       "ref_free": {o: [] for o in observer_scores_to_include}}
    subjects = {"full_ref": [],
               "ref_free": []}
    acquisition = {"full_ref": [],
                   "ref_free": []}
    niiFile = {"full_ref": [],
            "ref_free": []}

    data = np.loadtxt(input_csv, delimiter=',',
                      unpack=True, dtype=str)

    header = data[:, 0]
    if "Sbj" not in header:
        raise ValueError("No subject column found in the CSV file.")
    if "File" not in header:
        raise ValueError("No filename column found in the CSV file.")

    files = data[np.where(header == "File")[0][0], 1:]

    for i, file in enumerate(files):
        score = data[np.where(header == "Score")[0][0], i+1]
        if "nomoco" in file and score == "Not Found": # handle case where nomoco=ref
            pass
        else:
            is_reference = "ref" in file

            subject = data[np.where(header == "Sbj")[0][0], i+1] # Date
            acq = data[np.where(header == "Acq")[0][0], i+1] # Acq type (motion/no motion)
            
            subjects["ref_free"].append(subject)
            acquisition["ref_free"].append(acq)
            niiFile["ref_free"].append(file)
                    
            if not is_reference:
                subjects["full_ref"].append(subject)
                acquisition["full_ref"].append(acq)
                niiFile["full_ref"].append(file)

            for metric in full_reference_metrics:
                if metric not in header:
                    if i == 0:
                        print(f"No {metric} column found in the CSV file.")
                    metrics["full_ref"][metric].append(None)
                    continue
                if is_reference:
                    continue
                metrics["full_ref"][metric].append(
                    data[np.where(header == metric)[0][0], i+1].astype(float)
                )
            for metric in reference_free_metrics:
                if metric not in header:
                    if i == 0:
                        print(f"No {metric} column found in the CSV file.")
                    metrics["ref_free"][metric].append(None)
                    continue
                metrics["ref_free"][metric].append(
                    data[np.where(header == metric)[0][0], i+1].astype(float)
                )

            for observer in observer_scores_to_include:
                if observer not in header:
                    raise ValueError(f"No {observer} column found in the CSV file.")
                if not is_reference and score != "Not Found":
                    observer_scores["full_ref"][observer].append(
                        data[np.where(header == observer)[0][0], i+1].astype(float)
                    )
                observer_scores["ref_free"][observer].append(
                data[np.where(header == observer)[0][0], i+1].astype(float)
                )

    return subjects, acquisition, niiFile, metrics, observer_scores


def plot_correlation_heatmap(spearman_corr, original_metrics_order,
                             out_folder=None):
    # setting the colormap to be grey between -0.6 and 0.6
    coolwarm = plt.get_cmap('coolwarm')
    red_part = coolwarm(np.linspace(0.6, 1, 128))
    blue_part = coolwarm(np.linspace(0, 0.4, 128))
    grey = np.full((384, 4), 0.95)
    colors = np.vstack([blue_part, grey, red_part])
    cmap = mcolors.LinearSegmentedColormap.from_list('cmap', colors)

    # Filter correlations with p-value < 0.05
    filtered_corr = {
    metric: data["corr"]
    for metric, data in spearman_corr.items()
    if data["p_val"] < 0.05}
    
    # Create a matrix for the heatmap
    metrics = [metric for key in original_metrics_order
               for metric in original_metrics_order[key]]
    heatmap_data = np.zeros((len(metrics),1))

    for j, metric in enumerate(metrics):
        heatmap_data[j] = filtered_corr.get(metric, np.nan)

    fig_width = 3
    fig_height = 6

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    heatmap = sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap=cmap,
                          yticklabels=metrics,
                          xticklabels=False,
                          ax=ax, square=True,
                cbar_kws={"shrink": 1}, vmin=-1, vmax=1)
    ax.set_xlabel('Metrics', fontsize=14)
    # ax.set_title('Spearman Correlation Coefficients (p-value < 0.05)',fontsize=20)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=12)
    for text in heatmap.texts:
        text.set_size(9)
    cbar = heatmap.collections[0].colorbar
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label('$\\rho$', fontsize=14, rotation=0)
    plt.tight_layout()
    if out_folder is not None:
        plt.savefig(f'{out_folder}/correlation_heatmap.png', dpi=600)


def plot_scatter_plots(metrics, observer_scores, original_metrics_order,
                       out_folder=None):
    types = list(metrics.keys())
    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(25, 10))
    
    i = 0
    for type in types:
        for metric in original_metrics_order[type]:
            row = i // 5
            col = i % 5
            ax = axes[row, col]
            x = metrics[type][metric]
            if not any(element is None for element in x):
                y = observer_scores[type]["Score"]
                ax.scatter(x, y, alpha=0.7)
                ax.set_xlabel(f'{metric}', fontsize=20)
                ax.set_ylabel('Observer Scores', fontsize=20)
                ax.grid(True)

                # Calculate and plot regression line
                if len(x) > 1 and len(y) > 1:
                    slope, intercept = np.polyfit(x, y, 1)
                    ax.plot(x, slope * np.array(x) + intercept, color='tab:green')

                # Set x-ticks
                x_ticks = np.linspace(min(x), max(x), 4)
                ax.set_xticks(x_ticks)

                # Format x-tick labels
                ax.set_xticklabels([f'{tick:.1e}' if tick < 0.1 else f'{tick:.2f}' for tick in x_ticks])
            
            i += 1

        # plt.suptitle(f'Scatter Plots of Metrics vs. Observer Scores '
        #              f'for {sequence}', fontsize=18)
        plt.tight_layout()
        if out_folder is not None:
            plt.savefig(f'{out_folder}/scatter_plots_cubric.png', dpi=600)



def main():    
    scoredate = "2024-10-22_22-14"
    parser = argparse.ArgumentParser(
        description='Process CSV files to gather observer scores.')
    parser.add_argument(
        '--input_csv',
        help='Path to the CSV file containing the metrics and observer scores',
        default="./Results/CUBRICdata/" + scoredate + "/ImageQualityMetricsScores.csv")

    args = parser.parse_args()
    subjects, acquisition, file, metrics, observer_scores = load_metrics_scores(args.input_csv)
    out_dir = os.path.dirname(args.input_csv)

    # Calculate the spearman correlation coefficients between the metrics and observer scores:
    spearman_corr = {}
    for type in observer_scores.keys():
        for metric in metrics[type].keys():
            spearman_corr[metric] = {}
            if any(element is None for element in metrics[type][metric]):
                sp_corr, p_val = (np.nan, np.nan)
            else:               
                sp_corr, p_val = spearmanr(metrics[type][metric],
                                            observer_scores[type]["Score"])
            spearman_corr[metric]["corr"] = sp_corr
            spearman_corr[metric]["p_val"] = p_val

    original_metrics_order = {
        "full_ref": ["SSIM", "PSNR", "FSIM", "VIF", "LPIPS"],
        "ref_free": ["AES", "TG", "NGS", "GE", "IE"]
    }
    
    # plot_correlation_heatmap(spearman_corr, original_metrics_order, out_dir)
    
    with open(f"{out_dir}/correlation_coefficients.csv", "w",
    newline='') as file:
        writer = csv.writer(file)
        writer.writerow(
            ["Metric", "Correlation Coefficient", "P-Value"])
        for metric in original_metrics_order["full_ref"]:
            writer.writerow(
                [metric, spearman_corr[metric]["corr"],
                spearman_corr[metric]["p_val"]])
        for metric in original_metrics_order["ref_free"]:
            writer.writerow(
                [metric, spearman_corr[metric]["corr"],
                spearman_corr[metric]["p_val"]])

    
    # plot_scatter_plots(metrics, observer_scores, original_metrics_order, out_dir)
    

if __name__ == "__main__":
    main()

print("Done")


