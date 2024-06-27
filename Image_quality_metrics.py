
'''
Code used for the analysis of the ISMRM 2022 abstract "Evaluating the match 
of image quality metrics with radiological assessment in a dataset with and 
without motion artifacts" 

'''

import nibabel as nib
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from metrics.AES import aes
from metrics.fsim import calc_fsim
from metrics.gradient_metrics import *
from metrics.perceptual_metric import perceptual_metric


def Compute_Metric(filename, brainmask_file=False, ref_file=False, 
                   normal=True):
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
    
    # metrics_dict = {
    #     "full_reference": {
    #         "FSIM": calc_fsim,
    #         "PerceptualMetric": perceptual_metric},
    #     "reference_free": {
    #         "AES": aes,
    #         "Tenengrad": tenengrad,
    #         "NGS": normalized_gradient_squared,
    #         "GradientEntropy": gradient_entropy}
    # }
    
    metrics_dict = {
        "reference_free": {
            "AES": aes}
    }
    
    img = nib.load(filename).get_fdata().astype(np.uint16)
    
    if brainmask_file != False:
        brainmask = nib.load(brainmask_file).get_fdata().astype(np.uint16)
    else:
        brainmask = []
        
    if ref_file != False:
        ref = nib.load(ref_file).get_fdata().astype(np.uint16)
    
    res = []
    
    for m in metrics_dict["reference_free"]:        
        if m in metrics_dict["reference_free"]:
            if brainmask_file != False:
                img_masked = np.multiply(img, brainmask)
            else:
                img_masked = img
                            
            if normal == True:
                mean_img = np.mean(img_masked)
                std_img = np.std(img_masked)
                img_final = (img_masked-mean_img)/std_img
            else:
                img_final = img_masked
                
            metric_value = metrics_dict['reference_free'][m](img_final)
            print(f"{m}: {metric_value}")
            
            res = np.append(res,metric_value)

        else:
            raise NotImplementedError("Metric {} not implemented.".format(m))

    
    return res

