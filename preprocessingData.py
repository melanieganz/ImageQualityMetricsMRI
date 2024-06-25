import os
import pandas as pd
import subprocess
import nibabel as nib
import datalad

# Function to execute system command and check output
def execute_command(command):
    result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to execute command '{command}': {result.stderr.decode('utf-8')}")
    return result.stdout.decode('utf-8')

# Function to find all reference images in a directory for given sequences
def find_reference_images(directory, seq):
    reference_images = None 
    
    for filename in os.listdir(directory):
        if seq.lower() in filename.lower() and filename.endswith(".nii") and "run-01" in filename and "pmcoff" in filename:
            reference_images=os.path.join(directory, filename)
    return reference_images

# ------------------------------------------------------------------------------------------------------
# Pre-processing applied on data from:
# Melanie Ganz and Hannah Eichhorn (2022). Datasets with and without deliberate head movements for 
# evaluating the performance of markerless prospective motion correction and selective reacquisition 
# in a general clinical protocol for brain MRI. 
# ------------------------------------------------------------------------------------------------------
# The data is available for download at OpenNeuro. 
# ------------------------------------------------------------------------------------------------------
# Before downloading, make sure you have Datalad installed:
    # pip install datalad[full] (pip)
    # or
    # conda install -c conda-forge datalad (conda)
# And Git Annex: 
    # sudo apt-get install git-annex (Debian/Ubuntu)
    # or
    # brew install git-annex (Mac)
# ------------------------------------------------------------------------------------------------------
# This code requires FSL. 
    # https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation
# ------------------------------------------------------------------------------------------------------

# URL of the Git repository to clone
repo_url = "https://github.com/OpenNeuroDatasets/ds004332.git"

# Directory where you want to clone the repository
data_dir = "OpenNeuro_dataset"

# Check if the directory already exists
if os.path.exists(data_dir):
    print(f"The directory '{data_dir}' already exists.")
else:
    # Command to clone the repository
    clone_command = ["git", "clone", repo_url, data_dir]

    # Execute the command
    try:
        subprocess.run(clone_command, check=True)
        # Get the actual data        
        get_data = ["datalad","get","-r",data_dir] 
        subprocess.run(get_data, check=True)
        
        print("Repository cloned successfully")
    except subprocess.CalledProcessError as e:
        print(f"Error cloning repository: {e}")
        

# Define the sequences to look for in file names
sequences = ["mprage", "t1tirm", "t2tse", "flair"]

# Loop through each subject folder (sub-01, sub-02, ..., sub-22)
subject_folders = sorted(f for f in os.listdir(data_dir) if f.startswith("sub-"))
for subject_folder in subject_folders:
    anat_folder = os.path.join(data_dir, subject_folder, "anat")
    
    # Process each sequence type and its corresponding reference image
    for seq in sequences:
        ref_image = find_reference_images(anat_folder, seq)
        
        if ref_image:
            print(f"Found reference image for {subject_folder} ({seq}): {ref_image}")
            
            # Determine output directories for FLIRT
            sequence_output_dir = os.path.join(data_dir, subject_folder, seq.lower())
            os.makedirs(sequence_output_dir, exist_ok=True)
            
            if seq == "mprage":
                # For MPRAGE sequence, run BET if it is the reference
                mprage_bet = os.path.join(sequence_output_dir, "bet_mprage")
                
                bet_command = ['bet', ref_image, mprage_bet, '-R', '-f', '0.4', '-m']
                subprocess.run(bet_command, check=True)
                print(f"BET completed for {subject_folder} ({seq})")
                
                # Get the MPRAGE BET mask file
                mprage_bet_mask = os.path.join(sequence_output_dir, "bet_mprage_mask.nii.gz")
                
                # Register non-reference MPRAGE images to their corresponding MPRAGE references
                for filename in os.listdir(anat_folder):
                    if "mprage" in filename.lower() and filename.endswith(".nii") and filename != os.path.basename(ref_image):
                        input_image = os.path.join(anat_folder, filename)
                        flirt_output_dir = os.path.join(data_dir, subject_folder, seq.lower())
                        os.makedirs(flirt_output_dir, exist_ok=True)
                        
                        flirt_output = os.path.join(flirt_output_dir, f"align_{filename}")
                        
                        flirt_command = ["flirt", "-in", input_image, "-ref", ref_image, "-out", flirt_output,'-dof','6']
                        subprocess.run(flirt_command, check=True)
                        print(f"FLIRT completed for {filename} in {subject_folder} ({seq})")                               
                                        
            else:
                # For other sequences, register the MPRAGE BET mask to the reference image using FLIRT
                flirt_output_dir = os.path.join(data_dir, subject_folder, seq.lower())
                os.makedirs(flirt_output_dir, exist_ok=True)
                
                flirt_output = os.path.join(flirt_output_dir, f"align_{seq}_mask.nii.gz")
                flirt_command = ["flirt", "-in", mprage_bet_mask, "-ref", ref_image, "-out", flirt_output,'-dof','6']
                fslmath_command = ["fslmaths", flirt_output, '-thr', '0.1', '-bin', flirt_output]
                
                subprocess.run(flirt_command, check=True)
                subprocess.run(fslmath_command, check=True)
                print(f"FLIRT mask completed for {subject_folder} ({seq})")
    
                # Register non-reference images of other sequences to their corresponding references        
                for filename in os.listdir(anat_folder):
                    if filename.endswith(".nii") and filename != os.path.basename(ref_image) and seq in filename:
                        input_image = os.path.join(anat_folder, filename)
                        flirt_output_dir = os.path.join(data_dir, subject_folder, seq.lower())
                        os.makedirs(flirt_output_dir, exist_ok=True)
                        
                        flirt_output = os.path.join(flirt_output_dir, f"align_{filename}.nii.gz")
                        
                        flirt_command = ["flirt", "-in", input_image, "-ref", ref_image, "-out", flirt_output]
                        subprocess.run(flirt_command, check=True)
                        print(f"FLIRT completed for {filename} in {subject_folder} ({seq})")
                        
    print(f"Process completed for {subject_folder}")