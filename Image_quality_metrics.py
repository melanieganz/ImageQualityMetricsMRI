
'''
Code used for the analysis of the ISMRM 2022 abstract "Evaluating the match 
of image quality metrics with radiological assessment in a dataset with and 
without motion artifacts" 

'''

import nibabel as nib
import numpy as np
from metrics.AES import aes
from metrics.fsim import calc_fsim

def Compute_Metric(filename, metric, brainmask_file=False, ref_file=False, 
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
    
    metric_dict = {
        "full_reference": {
            "FSIM": calc_fsim},
        "reference_free": {
            "AES":aes}
    }
    
    img = nib.load(filename).get_fdata().astype(np.uint16)
    
    if brainmask_file != False:
        brainmask = nib.load(brainmask_file).get_fdata().astype(np.uint16)
    else:
        brainmask = []
        
    if ref_file != False:
        ref = nib.load(ref_file).get_fdata().astype(np.uint16)
    
    if metric == 'all':
        metrics = ['AES', 'FSIM']
    else:
        metrics = [metric]
    
    res = []
    for m in metrics:
        if m in ['FSIM']:
            if brainmask_file != False:
                brainmask_fl = brainmask.flatten()
                d_ref = ref.flatten()
                ref_ = d_ref[brainmask_fl>0]
                data_ref = ref_
                dat = img.flatten()
                img_ = dat[brainmask_fl>0]
                data_img = img_
            else:
                data_ref = ref
                data_img = img
            
            if normal == True:
                print('Values calculated on normalized images')
                mean_ref = np.mean(data_ref)
                std_ref = np.std(data_ref)
                data_ref = (data_ref-mean_ref)/std_ref
                
                mean_img = np.mean(data_img)
                std_img = np.std(data_img)
                data_img = (data_img-mean_img)/std_img
            
            peak = np.amax(data_ref)

            res.append(metric_dict[m](data_ref, data_img, 
                                            data_range=peak, 
                                            gaussian_weights=True))
        
        if m in ['AES']:
            if brainmask_file != False:
                brainmask_fl = brainmask.flatten()
                dat = img.flatten()
                img_ = dat[brainmask_fl>0]
                data_img = img_
                
            if normal == True:
                mean_img = np.mean(data_img)
                std_img = np.std(data_img)
                img_n = (img-mean_img)/std_img
            else:
                img_n = img_ 
                
            data_img = np.reshape(img_n,np.shape(img))
            res.append(metric_dict[m](data_img, brainmask))

    
    return res

