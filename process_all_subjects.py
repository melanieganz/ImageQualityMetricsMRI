"""
Script to estimate image quality metrics.

Run this script with:
nohup python -u process_all_subjects.py > Results/log_24_08_21.txt &
"""

import os
import shutil
import datetime
import subprocess
from match_metrics_scores import process_csv
from compute_metrics import compute_metrics
from data_utils import find_reference_images

debug = False
data_dir = "OpenNeuro_dataset"
out_dir = "Results/OpenNeuro/"
normalisation = True
mask_metric_values = True
reduction = "mean"

out_dir = out_dir + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M/")
os.makedirs(out_dir, exist_ok=True)

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
                    if seq =="mprage":
                        seq_bet_mask = os.path.join(seq_folder,
                                                    f"bet_{seq}_mask.nii.gz")
                    else:
                        seq_bet_mask = os.path.join(seq_folder,
                                                    f"align_{seq}_mask.nii.gz")
            
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
                        shutil.copyfile("helper_run_calculation.sh",
                                    "tmp_helper_run_calculation.sh")
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
                        with open("tmp_helper_run_calculation.sh", "a") as file:
                            file.write("\n" + command + "\n")

                        subprocess.run("bash tmp_helper_run_calculation.sh",
                                       shell=True)
                        os.remove("tmp_helper_run_calculation.sh")

    print(f"Process completed for {subject_folder}")

print("All subjects processed. Now matching observer scores...")
input_csv = f"{out_dir}/ImageQualityMetrics.csv"
output_csv = f"{out_dir}/ImageQualityMetricsScores.csv"
in_dir = "./observer_scores/"
process_csv(input_csv, output_csv, in_dir)
print("Process completed.")