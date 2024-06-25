import os
import nibabel as nib
import numpy as np
import pandas as pd

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
        ref_image = find_reference_images(seq_folder, seq)
        
        if ref_image:
            print(f"Found reference image for {subject_folder} ({seq}): {ref_image}")

            # For each file (reference incuded):
            for filename in os.listdir(seq_folder):
                if seq in filename.lower() and filename.endswith(".nii"):
                    
                    # Get the mask file
                    if seq =="mprage":
                        seq_bet_mask = os.path.join(seq_folder, "bet_{seq}_mask.nii.gz")
                    else:
                        seq_bet_mask = os.path.join(seq_folder, "align_{seq}_mask.nii.gz")
            
                    input_image = os.path.join(seq_folder, filename)
                    print(f"Input image is {input_image}")   
                    print(f"Mask is {seq_bet_mask}")    
                    print(f"Reference is {ref_image}")       
                    
                    # run metric calculation
                    imq = Compute_Metric(filename, "all", brainmask_file=seq_bet_mask, ref_file=ref_image, 
                                    normal=True)         
                    
                    res_imq = {'Sbj':subject_folder,
                                'File': filename,                          
                                'SSIM':imq[0], 
                                'PSNR':imq[1], 
                                'Tenengrad':imq[2],
                                'GradEntropy':imq[3],
                                'ImgEntropy':imq[4],
                                'AES':imq[5],
                                'CoEnt':imq[6]  
                                }      
                    results_list.append(res_imq)
                                                                                                                              
    print(f"Process completed for {subject_folder}")