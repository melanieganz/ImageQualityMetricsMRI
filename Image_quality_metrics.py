
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


def sort_out_zero_slices(img, ref):
    """ Only keep non-zero slices in img and ref. """

    zero_slices_img = np.where(np.sum(img, axis=(1, 2)) == 0)[0]
    if ref is not None:
        zero_slices_ref = np.where(np.sum(ref, axis=(1, 2)) == 0)[0]
        zero_slices = np.unique(np.concatenate((zero_slices_img, zero_slices_ref)))
        img = np.delete(img, zero_slices, axis=0)
        ref = np.delete(ref, zero_slices, axis=0)
    else:
        img = np.delete(img, zero_slices_img, axis=0)
        ref = None
    return img, ref


def compute_metrics(filename, brainmask_file=False, ref_file=False,
                    normal=True):
    """
    Calculate metrics for a given image.

    Parameters
    ----------
    filename : str
        filename of the nifti image which is supposed to be evaluated.
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

    """
    
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

    # Load data, brainmask and reference:
    img = nib.load(filename).get_fdata().astype(np.uint16)
    
    if brainmask_file:
        brainmask = nib.load(brainmask_file).get_fdata().astype(np.uint16)
    else:
        brainmask = []
        
    if ref_file:
        ref = nib.load(ref_file).get_fdata().astype(np.uint16)
    else:
        ref = None

    # Roll the z-axis to the first dimension if the sequence is t1tirm or t2tse
    if "t1tirm" in filename or "t2tse" in filename:
        img = np.rollaxis(img, 2)
        ref = (np.rollaxis(ref, 2) if ref is not None else None)
        if brainmask_file:
            brainmask = np.rollaxis(brainmask, 2)

    # Apply brainmask if available:
    if brainmask_file:
        img_masked = np.multiply(img, brainmask)
        ref_masked = (np.multiply(ref, brainmask) if ref is not None else None)
    else:
        img_masked = img
        ref_masked = ref

    # Sort out the slices with only zeros:
    img_masked, ref_masked = sort_out_zero_slices(img_masked, ref_masked)

    # Normalization:
    if normal:
        img = min_max_scale(img_masked)
        ref = (min_max_scale(ref_masked) if ref is not None else None)
    else:
        img = img_masked
        ref = ref_masked
    
    res = []
    for m in metrics_dict["full_reference"]:
        # Calculate metric: for SSIM and PSNR the data range must be set to 1
        if m == "SSIM" or m == "PSNR":
            metric_value = metrics_dict['full_reference'][m](img, ref,
                                                             data_range=1)
        else:
            metric_value = metrics_dict['full_reference'][m](img, ref,
                                                             reduction='worst')

        print(f"{m}: {metric_value}")
        res = np.append(res,metric_value)

    for m in metrics_dict["reference_free"]:
        # Calculate metric:
        metric_value = metrics_dict['reference_free'][m](img)

        print(f"{m}: {metric_value}")
        res = np.append(res,metric_value)

    return res

