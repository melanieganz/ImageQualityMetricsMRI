
'''
The code is based on:
McGee K, Manduca A, Felmlee J et al. Image metric-based correction 
(autocorrection) of motion effects: analysis of image metrics. J Magn Reson 
Imaging. 2000; 11(2):174-181
'''

import numpy as np
from scipy.stats import entropy


def imageEntropy(img, brainmask = None):
    '''
    Parameters
    ----------
    img : numpy array
        image for which the metrics should be calculated.
    brainmask : bool, optional
        If True, a brainmask was used to mask the images before
        calculating the metrics. Image is flattened prior metric
        estimation. The default is False.
        
    Returns
    -------
    ie : float
        Image Entropy of the input image.
    '''
    
    
    if brainmask is not None:
        image = img.flatten()
        image = image[image > 0]
    else:
        image = img.flatten()
    
    _, counts = np.unique(image, return_counts=True)
    ie = entropy(counts, base=2)
    
    return ie


