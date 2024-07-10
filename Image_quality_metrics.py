
'''
Code used for the analysis of the ISMRM 2022 abstract "Evaluating the match 
of image quality metrics with radiological assessment in a dataset with and 
without motion artifacts" 

'''

import numpy as np
import nibabel as nib
import warnings
warnings.filterwarnings("ignore")

from data_utils import *
from metrics.similarity_metrics import fsim
from metrics.perceptual_metrics import lpips
from metrics.information_metrics import vif
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from metrics.gradient_metrics import *
from metrics.ImageEntropy import imageEntropy
from metrics.AES import aes
from metrics.CoEnt import *


def Compute_Metric(filename, brainmask_file=False, ref_file=False, 
                   normal=True, permute=False):
    '''
    

    Parameters
    ----------
    filename : str
        filename of the nifti image which is supposed to be evaluated.
    metric : str
        which metric to calculate. Please choose between 'SSIM', 'PSNR', 
        'Tenengrad', 'AES', 'GradEntropy', 'ImgEntropy', 'CoEnt' and 'all'. 
        If the option 'all' is chosen, all metrics are returned in a list in 
        the order: SSIM, PSNR, Tenengrad, AES, GradEntropy, ImgEntropy, CoEnt.
    brainmask_file : str, optional.
        filename for the corresponding brainmask. If it is set to False, the
        metric will be calculated on the whole image.The default is False.
    ref_file : str, optional
        filename for the reference nifti scan which the image is supposed to 
        be compared to. This is only needed for SSIM and PSNR. The default is 
        False.
    normal : bool, optional
        whether the data should be normalized before metric calculation. The 
        default is True.

    Returns
    -------
    res : float
        value of the metric.

    '''
    
    metrics_dict = {
        "full_reference": {
            'SSIM':structural_similarity, 
            'PSNR':peak_signal_noise_ratio, 
            "FSIM": fsim,
            "VIF": vif,
            "PerceptualMetric": lpips},

        "reference_free": {
            "AES": aes,
            "Tenengrad": tenengrad,
            "NGS": normalized_gradient_squared,
            "GradientEntropy": gradient_entropy, 
            "Entropy": imageEntropy,
            "CoEnt": coent
            }
    }
    
    # Load data
    img = nib.load(filename).get_fdata().astype(np.uint16)
    
    # Load brainmask
    if brainmask_file != False:
        brainmask = nib.load(brainmask_file).get_fdata().astype(np.uint16)
    else:
        brainmask = []
    
    # Load reference        
    if ref_file != False:
        ref = nib.load(ref_file).get_fdata().astype(np.uint16)
        
    # Permute data
    if permute == True:
        img = np.transpose(img, (2,0,1))
        ref = np.transpose(ref, (2,0,1))
    if permute == True and brainmask_file != False:
        brainmask = np.transpose(brainmask, (2,0,1))

    # Apply brainmask
    if brainmask_file != False:
        img_masked = np.multiply(img, brainmask)
        ref_masked = np.multiply(ref, brainmask)
    else:
        img_masked = img
        ref_masked = ref

    # Normalization
    if normal == True:
        img = min_max_scale(img_masked)
        ref = min_max_scale(ref_masked)
    else:
        img = img_masked
        ref = ref_masked
            
    res = []
      
    for m in metrics_dict["full_reference"]:
        # Calculate metric: for SSIM and PSNR the data range must be set to 1
        if m == "SSIM" or m == "PSNR":
            metric_value = metrics_dict['full_reference'][m](img, ref, data_range=1.)            
        else:
            metric_value = metrics_dict['full_reference'][m](img, ref, reduction='worst')  
                        
        print(f"{m}: {metric_value}")
        
        res = np.append(res,metric_value)
    
    for m in metrics_dict["reference_free"]:
        if brainmask_file != False:
            img_masked = np.multiply(img, brainmask)
        else:
            img_masked = img
                        
        if normal == True and m != "CoEnt":
            img_final = min_max_scale(img_masked)
        else:
            img_final = img_masked
        
        # Calculate metric: 
        metric_value = metrics_dict['reference_free'][m](img_final)
            
        print(f"{m}: {metric_value}")
        
        res = np.append(res,metric_value)

    return res

