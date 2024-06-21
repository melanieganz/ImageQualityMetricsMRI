import os
import pandas as pd
import subprocess
import nibabel as nib

# Function to execute system command and check output
def execute_command(command):
    result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to execute command '{command}': {result.stderr.decode('utf-8')}")
    return result.stdout.decode('utf-8')

# Function to find all reference images in a directory for given sequences
def find_reference_images(directory, sequences):
    reference_images = {}
    for seq in sequences:
        reference_images[seq] = []
        for filename in os.listdir(directory):
            if seq.lower() in filename.lower() and filename.endswith(".nii"):
                reference_images[seq].append(os.path.join(directory, filename))
    return reference_images

# URL of the Git repository to clone
repo_url = "https://github.com/OpenNeuroDatasets/ds004332.git"

# Directory where you want to clone the repository
data_dir = "test"

# Check if the directory already exists
if os.path.exists(data_dir):
    print(f"The directory '{data_dir}' already exists.")
else:
    # Command to clone the repository
    clone_command = ["git", "clone", repo_url, data_dir]

    # Execute the command
    try:
        subprocess.run(clone_command, check=True)
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
                bet_output_dir = os.path.join(sequence_output_dir, "bet_output")
                os.makedirs(bet_output_dir, exist_ok=True)
                
                bet_command = ['bet', ref_image, bet_output_dir, '-R', '-f', '0.5', '-g', '0', '-m']
                subprocess.run(bet_command, check=True)
                print(f"BET completed for {subject_folder} ({seq})")
                
                # Get the MPRAGE BET mask file
                mprage_bet_mask = os.path.join(bet_output_dir, "mprage_mask.nii.gz")
                
                # Register non-reference MPRAGE images to their corresponding MPRAGE references
                for filename in os.listdir(anat_folder):
                    if "mprage" in filename.lower() and filename.endswith(".nii") and filename != os.path.basename(ref_image):
                        input_image = os.path.join(anat_folder, filename)
                        flirt_output_dir = os.path.join(data_dir, subject_folder, seq.lower())
                        os.makedirs(flirt_output_dir, exist_ok=True)
                        
                        flirt_output = os.path.join(flirt_output_dir, f"align_img.nii.gz")
                        
                        flirt_command = ["flirt", "-in", input_image, "-ref", ref_image, "-out", flirt_output, "-applyxfm"]
                        subprocess.run(flirt_command, check=True)
                        print(f"FLIRT completed for {filename} in {subject_folder} ({seq})")                               
                                        
            else:
                # For other sequences, register the MPRAGE BET mask to the reference image using FLIRT
                flirt_output_dir = os.path.join(sequence_output_dir, "flirt_output")
                os.makedirs(flirt_output_dir, exist_ok=True)
                
                flirt_output = os.path.join(flirt_output_dir, f"align_{seq}_mask.nii.gz")
                
                flirt_command = ["flirt", "-in", mprage_bet_mask, "-ref", ref_image, "-out", flirt_output, "-applyxfm"]
                subprocess.run(flirt_command, check=True)
                print(f"FLIRT mask completed for {subject_folder} ({seq})")
    
                # Register non-reference images of other sequences to their corresponding references        
                for filename in os.listdir(anat_folder):
                    if filename.endswith(".nii") and filename != os.path.basename(ref_image):
                        input_image = os.path.join(anat_folder, filename)
                        flirt_output_dir = os.path.join(data_dir, subject_folder, seq.lower())
                        os.makedirs(flirt_output_dir, exist_ok=True)
                        
                        flirt_output = os.path.join(flirt_output_dir, f"align_img.nii.gz")
                        
                        flirt_command = ["flirt", "-in", input_image, "-ref", ref_image, "-out", flirt_output, "-applyxfm"]
                        subprocess.run(flirt_command, check=True)
                        print(f"FLIRT completed for {filename} in {subject_folder} ({seq})")