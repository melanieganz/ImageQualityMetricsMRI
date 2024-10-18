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
from match_metrics_scores import process_csv_cubric
from compute_metrics import compute_metrics
from data_utils import find_reference_images

debug = False
data_dir = "/Users/emarchetto/Library/CloudStorage/OneDrive-NYULangoneHealth/ImageQualityMetrics_ISMRM24/CUBRIC_Data/FLIRT/"
mask_dir = "/Users/emarchetto/Library/CloudStorage/OneDrive-NYULangoneHealth/ImageQualityMetrics_ISMRM24/CUBRIC_Data/mask/"
out_dir = "Results/CUBRICdata/"


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


results_list = []

# Structure is: main folder (dates) with multiple subfolders (mprage with/without motion)
subject_folders = sorted(f for f in os.listdir(data_dir) if not f.startswith('.'))
for subject_folder in subject_folders:   
    acquisitions = sorted(f for f in os.listdir(os.path.join(data_dir, subject_folder)) if not f.startswith('.'))
    
    for acq in acquisitions:
        acq_folder = os.path.join(data_dir, subject_folder, acq)     

        # Each subfolder had the flirt_ref.nii.gz file saved (the same for each acquisition date)
        ref_image = os.path.join(acq_folder, f"flirt_ref.nii.gz")

        # Get the mask file
        mask_folder = os.path.join(mask_dir,subject_folder)

        # For each file (reference incuded):
        for filename in os.listdir(acq_folder):                    
            if apply_brainmask:
                acq_bet_mask = os.path.join(mask_folder,
                                                f"align_MPR_mask.nii.gz")
            else:
                acq_bet_mask = "none"

            input_image = os.path.join(acq_folder, filename)
            print(f"Input image is {input_image}")   
            print(f"Mask is {acq_bet_mask}")    
            print(f"Reference is {ref_image}")       
            
            # run metric calculation
            if debug:
                imq = compute_metrics(input_image,
                                        subject_folder,
                                        acq,
                                        f"{out_dir}/ImageQualityMetrics.csv",
                                        brainmask_file=acq_bet_mask,
                                        ref_file=ref_image, normal=normalisation,
                                        mask_metric_values=mask_metric_values,
                                        reduction=reduction)
            else:
                shutil.copyfile("helper_run_calculation.sh",
                            f"tmp_helper_run_calculation_{date_stamp}.sh")
                command = (
                    'python -u compute_metrics.py {} {} {} {}'
                    '/ImageQualityMetrics.csv {} {} --normal {} '
                    '--mask_metric_values {} --reduction {}'
                ).format(
                    input_image,
                    subject_folder,
                    acq,
                    out_dir,
                    acq_bet_mask,
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

input_csv = "./Results/CUBRICdata/" + out_dir + "/ImageQualityMetrics.csv"
output_csv = "./Results/CUBRICdata/" + out_dir + "/ImageQualityMetricsScores.csv"
scores_csv = "./observer_scores/CUBRIC_scores.csv"

process_csv_cubric(input_csv, output_csv, scores_csv)
print("Process completed.")
