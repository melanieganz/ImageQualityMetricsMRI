import os
import nibabel as nib
import numpy as np
import pandas as pd

import subprocess
from os import listdir
from os.path import join

from Image_quality_metrics import Compute_Metric

data_dir = "OpenNeuro_dataset"

# Function to find all reference images in a directory for given sequences
def find_reference_images(directory, seq):
    reference_images = None 
    
    for filename in os.listdir(directory):
        if seq.lower() in filename.lower() and filename.endswith(".nii") and "run-01" in filename and "pmcoff" in filename:
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
                if seq in filename.lower() and filename.endswith(".nii") or filename.endswith(".gz"):
                    
                    # Get the mask file
                    if seq =="mprage":
                        seq_bet_mask = os.path.join(seq_folder, f"bet_{seq}_mask.nii.gz")
                    else:
                        seq_bet_mask = os.path.join(seq_folder, f"align_{seq}_mask.nii.gz")
            
                    input_image = os.path.join(seq_folder, filename)
                    print(f"Input image is {input_image}")   
                    print(f"Mask is {seq_bet_mask}")    
                    print(f"Reference is {ref_image}")       
                    
                    # run metric calculation
                    imq = Compute_Metric(input_image, brainmask_file=seq_bet_mask, ref_file=ref_image, normal=True)      
                    
                    res_imq = {'Sbj':subject_folder,
                                'File': filename, 
                                'AES':imq[0],
                                'Tenengrad':imq[1],
                                'NGS':imq[2],
                                'Gradient Entropy':imq[3],
                                'Entropy':imq[4],
                                'CoEnt':imq[5],
                                'SSIM':imq[6],
                                'PSNR':imq[7],
                                'FSIM':imq[8],
                                'VIF':imq[9],
                                'PerceptualMetric':imq[10],
                                }      
                    results_list.append(res_imq)
                                                                                                                              
    print(f"Process completed for {subject_folder}")
    
# Save results in a csv file
results_df = pd.DataFrame(results_list)
results_df.to_csv("ImageQualityMetrics.csv", index=False)
