import numpy as np
import argparse
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns


def load_metrics_scores(input_csv):
    metrics_to_include = ["SSIM", "PSNR", "FSIM", "VIF", "LPIPS", "AES", "TG",
                          "NGS", "GE", "IE"]
    observer_scores_to_include = ["Radiographer1", "Radiographer2",
                                  "Neuroradiologist"]
    sequences = ["mprage", "t1tirm", "flair", "t2tse"]
    metrics = {
        s : {m: [] for m in metrics_to_include} for s in sequences
    }
    observer_scores = {
        s: {o: [] for o in observer_scores_to_include} for s in sequences
    }
    subjects = {
        s: [] for s in sequences
    }
    acquisition = {
        s: [] for s in sequences
    }

    data = np.loadtxt(input_csv, delimiter=',',
                      unpack=True, dtype=str)

    header = data[:, 0]
    if "Sbj" not in header:
        raise ValueError("No subject column found in the CSV file.")
    if "File" not in header:
        raise ValueError("No filename column found in the CSV file.")

    files = data[np.where(header == "File")[0][0], 1:]

    for i, file in enumerate(files):
        sequence = None
        for seq in sequences:
            if seq in file:
                sequence = seq
                break
        if sequence is None:
            raise ValueError(f"Sequence not recognized for file {file}")

        subject = data[np.where(header == "Sbj")[0][0], i+1]
        subjects[sequence].append(subject)
        acquisition[sequence].append(file)

        for metric in metrics_to_include:
            if metric not in header:
                raise ValueError(f"No {metric} column found in the CSV file.")
            metrics[sequence][metric].append(
                data[np.where(header == metric)[0][0], i+1].astype(float)
            )

        for observer in observer_scores_to_include:
            if observer not in header:
                raise ValueError(f"No {observer} column found in the CSV file.")
            observer_scores[sequence][observer].append(
                data[np.where(header == observer)[0][0], i+1].astype(float)
            )

    return subjects, acquisition, metrics, observer_scores


def plot_correlation_heatmap(spearman_corr, original_metrics_order):
    # Filter correlations with p-value < 0.05
    filtered_corr = {
        seq: {metric: data["corr"] for metric, data in metrics.items()
              if data["p_val"] < 0.05}
        for seq, metrics in spearman_corr.items()
    }

    # Create a matrix for the heatmap
    sequences = list(filtered_corr.keys())
    metrics = original_metrics_order
    heatmap_data = np.zeros((len(sequences), len(metrics)))

    for i, sequence in enumerate(sequences):
        for j, metric in enumerate(metrics):
            heatmap_data[i, j] = filtered_corr[sequence].get(metric, np.nan)

    # Plotting
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="coolwarm",
                xticklabels=metrics, yticklabels=sequences, ax=ax, square=True,
                cbar_kws={"shrink": 0.5})
    ax.set_xlabel('Metrics', fontsize=14)
    ax.set_ylabel('Sequences', fontsize=14)
    ax.set_title('Spearman Correlation Coefficients (p-value < 0.05)',
                 fontsize=16)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=12)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=12)
    plt.show()


def plot_scatter_plots(metrics, observer_scores, original_metrics_order):
    for sequence in metrics.keys():
        num_metrics = len(original_metrics_order)
        fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(20, 15))

        for i, metric in enumerate(original_metrics_order):
            row = i // 4
            col = i % 4
            ax = axes[row, col]
            x = metrics[sequence][metric]
            y = observer_scores[sequence]["Combined"]
            ax.scatter(x, y, alpha=0.7)
            ax.set_xlabel(f'{metric}', fontsize=16)
            ax.set_ylabel('Combined Observer Scores', fontsize=16)
            ax.grid(True)

            # Calculate and plot regression line
            if len(x) > 1 and len(y) > 1:
                slope, intercept = np.polyfit(x, y, 1)
                ax.plot(x, slope * np.array(x) + intercept, color='tab:green')

        # Hide any unused subplots
        for j in range(i + 1, 12):
            row = j // 4
            col = j % 4
            fig.delaxes(axes[row, col])

        plt.suptitle(f'Scatter Plots of Metrics vs. Combined Observer Scores '
                     f'for {sequence}', fontsize=18)
        plt.tight_layout()
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Process CSV files to gather observer scores.')
    parser.add_argument(
        '--input_csv',
        help='Path to the CSV file containing the metrics and observer scores',
        default="/home/iml/hannah.eichhorn/Results/ImageQualityMetrics/"
                "OpenNeuro/2024-08-13_16-45/ImageQualityMetricsScores.csv"
    )

    args = parser.parse_args()
    subjects, acquisition, metrics, observer_scores = load_metrics_scores(args.input_csv)

    # Combine observer scores with double weighting for neuroradiologist
    for sequence in observer_scores.keys():
        observer_scores[sequence]["Combined"] = np.mean(
            [observer_scores[sequence]["Radiographer1"],
             observer_scores[sequence]["Radiographer2"],
             observer_scores[sequence]["Neuroradiologist"],
             observer_scores[sequence]["Neuroradiologist"]],
            axis=0
        )

    # Calculate the spearman correlation coefficients between the metrics and observer scores:
    spearman_corr = {}
    for sequence in observer_scores.keys():
        spearman_corr[sequence] = {}
        for metric in metrics[sequence].keys():
            spearman_corr[sequence][metric] = {}
            sp_corr, p_val = spearmanr(metrics[sequence][metric],
                                       observer_scores[sequence]["Combined"])
            spearman_corr[sequence][metric]["corr"] = sp_corr
            spearman_corr[sequence][metric]["p_val"] = p_val

    original_metrics_order = ["SSIM", "PSNR", "FSIM", "VIF", "LPIPS", "AES", "TG", "NGS", "GE", "IE"]
    plot_correlation_heatmap(spearman_corr, original_metrics_order)

    # print mean correlation coefficient for each metric over all sequences
    for metric in original_metrics_order:
        if all([spearman_corr[seq][metric]["p_val"] < 0.05
                for seq in observer_scores.keys()]):
            mean_corr = np.mean([spearman_corr[seq][metric]["corr"]
                                 for seq in observer_scores.keys()])
            print(f"Mean correlation coefficient for {metric}: {mean_corr}")

    plot_scatter_plots(metrics, observer_scores, original_metrics_order)



if __name__ == "__main__":
    main()

print("Done")


