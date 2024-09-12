import os
import csv


def read_txt_files(directory, sequence):
    scores = {}
    filename = f"{sequence} Score.txt"

    with open(os.path.join(directory, filename), 'r') as file:
        for line in file:
            if line.startswith('#'):
                continue  # Skip header lines
            parts = line.strip().split()
            subject = parts[0]
            acquisition = parts[1]
            score1, score2, score3 = map(int, parts[2:])
            key = (subject, acquisition)
            scores[key] = [score1, score2, score3]
    return scores

def read_csv_file(filepath, sequence):
    scores = {}
    with open(filepath, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            if sequence in row[1]:
                subject = row[0]
                acquisition = row[1].split(sequence)[0][3:]
                if "RR" in row[1]:
                    acquisition += "RR_"
                score = int(row[2])
                key = (subject, acquisition)
                scores[key] = [score]
    return scores

def merge_scores(txt_scores, csv_scores):
    merged_scores = txt_scores.copy()
    for key, score_list in csv_scores.items():
        if key in merged_scores:
            merged_scores[key].append(score_list[0])
        else:
            merged_scores[key] = score_list
    return merged_scores

def match_acq_descr(tag, sequence):
    abbr_seq = {"T1_MPR": "mprage", "T2_TSE": "t2tse",
                "T2_FLAIR": "flair", "T1_TIRM": "t1tirm"}

    if tag == "MOCO_OFF_STILL_":
        true_tag = "ref_" + abbr_seq[sequence]
    else:
        true_tag = "acq-" + abbr_seq[sequence]
        if "MOCO_OFF" in tag:
            true_tag += "pmcoff_"
        elif "MOCO_ON" in tag:
            true_tag += "pmcon_"
        if "RR" in tag or "STILL" in tag:
            true_tag += "rec-wore_"
        else:
            true_tag += "rec-wre_"
        if "STILL" in tag:
            true_tag += "run-01"
        elif "NOD" in tag:
            true_tag += "run-02"
        elif "SHAKE" in tag:
            true_tag += "run-03"
    return true_tag

def sort_merged_scores(merged_scores):
    return sorted(merged_scores.items(), key=lambda item: item[0][0])

def write_merged_scores(directory, merged_scores, sequence):
    filename = os.path.join(directory, f"{sequence}_scores.csv")
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Subject', 'Acquisition', 'Radiographer1',
                         'Radiographer2', 'Radiologist1', 'Radiologist2'])
        sorted_scores = sort_merged_scores(merged_scores)
        for key, scores in sorted_scores:
            subject, acquisition = key
            acquisition = match_acq_descr(acquisition, sequence)
            subject = subject.replace('Subject_', 'sub-')
            writer.writerow([subject, acquisition] + scores)


# Paths to the directories and files
txt_directory = './observer_scores/original/'
csv_filepath = './observer_scores/original/radiologist_scores_nru.csv'
output_directory = './observer_scores/'
os.makedirs(output_directory, exist_ok=True)

sequences = ["T1_MPR", "T2_TSE", "T2_FLAIR", "T1_TIRM"]

# Read scores from txt files and csv file
for sequence in sequences:
    txt_scores = read_txt_files(txt_directory, sequence)
    csv_scores = read_csv_file(csv_filepath, sequence)

    # Merge the scores
    merged_scores = merge_scores(txt_scores, csv_scores)

    # Write the merged scores to new text files
    write_merged_scores(output_directory, merged_scores, sequence)

print("Done")