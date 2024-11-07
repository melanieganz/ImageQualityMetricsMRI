import glob
import numpy as np
import nibabel as nib
import subprocess
import os


def find_reference_images(directory, seq):
    reference_images = None

    for filename in os.listdir(directory):
        if (seq.lower() in filename.lower() and filename.endswith(".nii")
                and "run-01" in filename and "pmcoff" in filename):
            reference_images = os.path.join(directory, filename)
    return reference_images


def load_data(subject_folder, acq, rec, run):
    """ Load a specific acquisition of a subject."""

    filename = glob.glob(
        "{}/**acq-{}_rec-{}_run-{}**.nii".format(subject_folder, acq,
                                                 rec, run)
    )[0]

    return nib.load(filename).get_fdata().astype(np.uint16)


def sort_out_zero_slices(img, ref, brainmask=None):
    """ Only keep slices with more than 10% non-zero values in img and ref. """

    zero_slices_img = np.where(np.sum(img > 0, axis=(1, 2)) / img[0].size < 0.1)[0]

    if ref is not None:
        zero_slices_ref = np.where(np.sum(ref > 0, axis=(1, 2)) / ref[0].size < 0.1)[0]
        zero_slices = np.unique(np.concatenate((zero_slices_img, zero_slices_ref)))
        ref = np.delete(ref, zero_slices, axis=0)
    else:
        zero_slices = zero_slices_img

    img = np.delete(img, zero_slices, axis=0)
    if brainmask is not None:
        brainmask = np.delete(brainmask, zero_slices, axis=0)

    return img, ref, brainmask


def min_max_scale(img):
    """ Rescale image between [0,1] using the min/max method """
    min_val = np.min(img)
    max_val = np.max(img)
    img_scale = (img - min_val) / (max_val - min_val)
    return img_scale


def normalize_mean_std(img):
    """ Normalization to mean=0, std=1 """
    mean_img = np.mean(img)
    std_img = np.std(img)
    img_norm = (img-mean_img)/std_img
    return img_norm

def normalize_percentile(img, lower_percentile=1, upper_percentile=99.9, clip=True):
    """ Normalization to the lower and upper percentiles """
    img = img.astype(np.float32)
    lower = np.percentile(img, lower_percentile)
    upper = np.percentile(img, upper_percentile)
    img = (img - lower) / (upper - lower)
    if clip:
        img = np.clip(img, 0, 1)
    return img

def crop_img(img):
    '''
    Parameters
    ----------
    img : numpy array
        Image to be cropped.
    
    Returns
    -------
    crop_img : numpy array
        Cropped image such that all slices 
        contain at least one non-zero entry
    '''
    #Indices where the img is non-zero
    indices = np.array(np.where(img>0))
    
    #Max and Min x values where img is non-zero
    xmin = np.min(indices[0])
    xmax = np.max(indices[0])
    #Max and Min y values where img is non-zero
    ymin = np.min(indices[1])
    ymax = np.max(indices[1])
    #Max and Min z values where img is non-zero
    zmin = np.min(indices[2])
    zmax = np.max(indices[2])
    
    #Return cropped img
    return img[xmin:xmax, ymin:ymax , zmin:zmax]


def bin_img(img, n_levels = 128):
    '''
    Parameters
    ----------
    img : numpy array
        Image to bin.
    n_levels : int
        Number of levels to bin the intensities in
    
    Returns
    -------
    binned_img : numpy array
        Binned image, which has n_levels different 
        intensity values
    '''
    
    #Intensity values to map to
    vals, bins = np.histogram(img, bins = n_levels)

    #Bin image
    binned_img = bins[np.digitize(img, bins, right = True)]
    
    #Return binned image
    return binned_img


def is_dicom(filepath):
    """
    Given a file check if it is of dicom format
    
    Parameters
    ----------
    filepath : str
        filepath for the file that should be checked
    
    Returns
    -------
        True if it is a dicom False if not
    """
    #Split the filepath
    lst = filepath.split(".")
    #If the file ends with IMA r DCM return True
    if lst[-1].lower() == "ima" or lst[-1].lower() == "dcm":
        return True
    else: return False

def dicom2nifti(patient_id, dicom_directory, nifti_directory):
    '''
    Converts dicom image to nifti using
    FreeSurfers mri_convert
    
    
    Parameters
    ----------
    patient_id : str
        Name of folder containing the patients dicom files
    dicom_directory : str
        Filepath for the folder containing all dicom folders
    nifti_directory : str
        Filepath for the nifti directory
        
    Returns
    -------
    in_volume : str
        Filepath to the first dicom file
    out_volume : str
        Filepath to the nifti file
    '''
    
    #List of all dicom folders for the patient
    for dicom_fold in glob.glob(patient_id+"*/"):
        #First file in the dicom folder
        in_volume = glob.glob(dicom_fold+"*")[0]
        #Check if the file is dicom format
        if is_dicom(in_volume):
            #Output volume
            out_volume = nifti_directory+patient_id[len(dicom_directory):]+ dicom_fold[len(patient_id):-1]+".nii"
            
            print(in_volume)
            print(out_volume)
            print("----------")
            print()
            #Convert to nifti with mri_convert
            subprocess.run('mri_convert ' + in_volume + ' ' + out_volume+' --no-dwi', shell=True)
            

            
def convert_all(dicom_directory, nifti_directory):
    '''
    Converts all dicom images in a folder to nifti using
    FreeSurfers mri_convert
    
    
    Parameters
    ----------
    dicom_directory : str
        Filepath for the folder containing all dicom folders
    nifti_directory : str
        Filepath for the nifti directory
        where niftis will be saved to
        
    Returns
    -------
    in_volume : str
        Filepath to the first dicom file
    out_volume : str
        Filepath to the nifti file
    
    
    Example
    Consider two folders
    DICOMS and nifti_fold
    where DICOMS has folders, [John, Jane]
    each containing a folder for each dicom scan

    let dicom_dir be the path to the folder DICOMS 
    and nifti_dir the path to the folder nifti
    
    convert_all(dicom_directory, nifti_directory)
    will take each dicom image for both John and Jane 
    convert to nifti and save the nifti to the folder nifti_fold
    with the same name as the dicom filename + name, 
    eg T1_MPR_3D_SAG_P2_john.nii

    '''
    for patient in glob.glob(dicom_directory+"*/"):
        dicom2nifti(patient, dicom_directory, nifti_directory)
