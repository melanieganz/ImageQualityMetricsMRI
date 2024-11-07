"""
Script to estimate image quality metrics.

Run this script with:
nohup python -u process_all_subjects.py > Results/log_24_08_21.txt &
"""

import os
import shutil
import datetime
import subprocess
import argparse
from utils.match_metrics_scores import process_csv
from utils.compute_metrics import compute_metrics
from utils.data_utils import find_reference_images

debug = False
data_dir = "OpenNeuro_dataset"
out_dir = "Results/OpenNeuro/"


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser(description="Process all subjects to estimate "
                                             "image quality metrics.")
parser.add_argument("--normalisation", type=str,
                    default="percentile",
                    help="Normalisation method (default: 'percentile').")
parser.add_argument("--mask_metric_values", type=str2bool,
                    default=True, help="Whether to mask metric values (default: True).")
parser.add_argument("--reduction", type=str, default="worst",
                    help="Reduction method for metric calculation (default: 'worst').")
parser.add_argument("--apply_brainmask", type=str2bool,
                    default=True, help="Whether to apply brainmask (default: True).")
args = parser.parse_args()

normalisation = args.normalisation
mask_metric_values = args.mask_metric_values
reduction = args.reduction
apply_brainmask = args.apply_brainmask

if normalisation == "mean_std":
    print("mean_std normalisation is not applicable to all metrics.")

date_stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
out_dir = out_dir + date_stamp
os.makedirs(out_dir, exist_ok=True)

with open(f"{out_dir}/settings.txt", "w") as file:
    file.write(f"Normalisation: {normalisation}\n")
    file.write(f"Apply brainmask: {apply_brainmask}\n")
    file.write(f"Mask metric values: {mask_metric_values}\n")
    file.write(f"Reduction: {reduction}\n")

# Define the sequences to look for in file names
sequences = ["mprage", "t1tirm", "t2tse", "flair"]
results_list = []

# Loop through each subject folder (sub-01, sub-02, ..., sub-22)
subject_folders = sorted(f for f in os.listdir(data_dir) if f.startswith("sub-"))
for subject_folder in subject_folders:   

    for seq in sequences:
        seq_folder = os.path.join(data_dir, subject_folder, seq)
        
        # Find reference for that sequence (if available)
        ref_folder = os.path.join(data_dir, subject_folder, "anat")
        ref_temp = find_reference_images(ref_folder, seq)        
        
        if ref_temp:
            print(f"Found reference image for {subject_folder} ({seq}): {ref_temp}")
            
            # Command to copy file using copy command on Windows
            ref_image = os.path.join(seq_folder, f"ref_{seq}_image.nii")
            command = ['cp', '-f', ref_temp, ref_image]
            subprocess.run(command, check=True, shell=False)

            # For each file (reference incuded):
            for filename in os.listdir(seq_folder):                
                if (seq.lower() in filename.lower()
                        and filename.endswith((".nii", ".gz"))
                        and "mask" not in filename and "bet" not in filename):
                    
                    # Get the mask file
                    if apply_brainmask:
                        if seq =="mprage":
                            seq_bet_mask = os.path.join(seq_folder,
                                                        f"bet_{seq}_mask.nii.gz")
                        else:
                            seq_bet_mask = os.path.join(seq_folder,
                                                        f"align_{seq}_mask.nii.gz")
                    else:
                        seq_bet_mask = "none"
            
                    input_image = os.path.join(seq_folder, filename)
                    print(f"Input image is {input_image}")   
                    print(f"Mask is {seq_bet_mask}")    
                    print(f"Reference is {ref_image}")       
                    
                    # run metric calculation
                    if debug:
                        imq = compute_metrics(input_image,
                                              subject_folder,
                                              f"{out_dir}/ImageQualityMetrics.csv",
                                              brainmask_file=seq_bet_mask,
                                              ref_file=ref_image, normal=normalisation,
                                              mask_metric_values=mask_metric_values,
                                              reduction=reduction)
                    else:
                        shutil.copyfile("utils/helper_run_calculation.sh",
                                    f"tmp_helper_run_calculation_{date_stamp}.sh")
                        command = (
                            'python -u compute_metrics.py {} {} {}'
                            '/ImageQualityMetrics.csv {} {} --normal {} '
                            '--mask_metric_values {} --reduction {}'
                        ).format(
                            input_image,
                            subject_folder,
                            out_dir,
                            seq_bet_mask,
                            ref_image,
                            normalisation,
                            mask_metric_values,
                            reduction
                        )
                        with open(f"tmp_helper_run_calculation_{date_stamp}.sh",
                                  "a") as file:
                            file.write("\n" + command + "\n")

                        subprocess.run(f"bash tmp_helper_run_"
                                       f"calculation_{date_stamp}.sh",
                                       shell=True)
                        os.remove(f"tmp_helper_run_calculation_{date_stamp}.sh")

    print(f"Process completed for {subject_folder}")

print("All subjects processed. Now matching observer scores...")
input_csv = f"{out_dir}/ImageQualityMetrics.csv"
output_csv = f"{out_dir}/ImageQualityMetricsScores.csv"
in_dir = "./observer_scores/"
process_csv(input_csv, output_csv, in_dir)
print("Process completed.")
