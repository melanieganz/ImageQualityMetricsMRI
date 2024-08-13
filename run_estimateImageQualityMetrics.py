import os
import nibabel as nib
import numpy as np
import pandas as pd
import datetime
import subprocess
from os import listdir
from os.path import join

from Image_quality_metrics import compute_metrics

data_dir = "OpenNeuro_dataset"
out_dir = "Results/OpenNeuro/"
out_dir = out_dir + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M/")
os.makedirs(out_dir, exist_ok=True)

# Function to find all reference images in a directory for given sequences
def find_reference_images(directory, seq):
    reference_images = None 
    
    for filename in os.listdir(directory):
        if (seq.lower() in filename.lower() and filename.endswith(".nii")
                and "run-01" in filename and "pmcoff" in filename):
            reference_images=os.path.join(directory, filename)
    return reference_images


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
                    imq = compute_metrics(input_image,
                                          brainmask_file=seq_bet_mask,
                                          ref_file=ref_image, normal=True,
                                          mask_metric_values=True,
                                          reduction="worst")
                    
                    res_imq = {'Sbj': subject_folder,
                               'File': filename,
                               'SSIM': imq[0],
                               'PSNR': imq[1],
                               'FSIM': imq[2],
                               'VIF': imq[3],
                               'LPIPS': imq[4],
                               'AES': imq[5],
                               'TG': imq[6],
                               'NGS': imq[7],
                               'GE': imq[8],
                               'IE': imq[9]
                               }
                    results_list.append(res_imq)
                                                                                                                              
    print(f"Process completed for {subject_folder}")
    
# Save results in a csv file
results_df = pd.DataFrame(results_list)
results_df.to_csv(f"{out_dir}/ImageQualityMetrics.csv", index=False)
