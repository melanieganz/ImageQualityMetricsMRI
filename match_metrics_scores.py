import argparse
import csv

# Mapping of sequence types to their respective score files
sequence_to_file = {
    'mprage': 'T1_MPR Score.txt',
    't1tirm': 'T1_TIRM Score.txt',
    'flair': 'T2_FLAIR Score.txt',
    't2tse': 'T2_TSE Score.txt'
}


def match_acq_descr(filename):

    if "ref" in filename:
        tag = "MOCO_OFF_STILL_"
    else:
        if "pmcoff" in filename:
            tag = "MOCO_OFF_"
        elif "pmcon" in filename:
            tag = "MOCO_ON_"
        else:
            tag = "N/A"
        if "run-01" in filename:
            tag += "STILL_"
        elif "run-02" in filename:
            tag += "NOD_"
        elif "run-03" in filename:
            tag += "SHAKE_"
        if "rec-wre" in filename:
            tag += "RR_"

    return tag


# Function to parse the observer scores from the respective text file
def get_observer_scores(subject_id, sequence, in_dir, filename):
    sequence_file = in_dir + sequence_to_file.get(sequence, None)
    if not sequence_file:
        return None

    tag = match_acq_descr(filename)

    with open(sequence_file, 'r') as file:
        for line in file:
            if line.startswith('#'):
                continue  # Skip header lines
            parts = line.split()
            if parts[0] == subject_id.replace('sub-', 'Subject_'):
                if parts[1] == tag:
                    return parts[2:5]  # Return the three scores


# Function to determine the sequence type from the filename
def get_sequence_type(filename):
    if 'mprage' in filename:
        return 'mprage'
    elif 't1tirm' in filename:
        return 't1tirm'
    elif 'flair' in filename:
        return 'flair'
    elif 't2tse' in filename:
        return 't2tse'
    return None


# Main function to process the CSV file and gather observer scores
def process_csv(input_csv, output_csv, in_dir):
    with open(input_csv, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        header = next(reader)  # Skip the header line

        results = []
        for row in reader:
            tmp = row[0].split(',')
            subject_id = tmp[0]
            filename = tmp[1]
            sequence_type = get_sequence_type(filename)
            scores = get_observer_scores(subject_id, sequence_type, in_dir, filename)
            if scores:
                results.append(tmp + scores)
            else:
                results.append(tmp + ['N/A', 'N/A', 'N/A'])  # If no scores are found

    # Output results to a new CSV file
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(header[0].split(',') + ['Radiographer1', 'Radiographer2', 'Neuroradiologist'])
        writer.writerows(results)

    print(f"Processed data has been saved to {output_csv}")


def main():
    parser = argparse.ArgumentParser(
        description='Process CSV files to gather observer scores.')
    parser.add_argument(
        '--input_csv', help='Path to the input CSV file',
        default="/home/iml/hannah.eichhorn/Results/ImageQualityMetrics/OpenNeuro/2024-08-13_16-45/"
                "ImageQualityMetrics.csv"
    )
    parser.add_argument(
        '--output_csv', help='Path to the output CSV file',
        default="/home/iml/hannah.eichhorn/Results/ImageQualityMetrics/OpenNeuro/2024-08-13_16-45/"
                "ImageQualityMetricsScores.csv"
    )
    parser.add_argument(
        '--in_dir', help='Directory containing observer score files',
        default="./observer_scores/"
    )

    args = parser.parse_args()

    process_csv(args.input_csv, args.output_csv, args.in_dir)

if __name__ == "__main__":
    main()
