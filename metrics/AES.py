"""
Simon Chemnitz-Thomsen's code to calculate the metric Average Edge Strength.
Modified by Elisa Marchetto to remove zeros from the masking process from the image


Code is based on the article:
Quantitative framework for prospective motion correction evaluation
Nicolas Pannetier, Theano Stavrinos, Peter Ng, Michael Herbst,
Maxim Zaitsev, Karl Young, Gerald Matson, and Norbert Schuff
"""

import numpy as np
from skimage.feature import canny
from scipy.ndimage import convolve
from data_utils import crop_img, bin_img


def aes(img, brainmask=None, sigma=np.sqrt(2), n_levels=128, bin=False,
        crop=True, weigt_avg=False, reduction='mean'):
    """
    Parameters
    ----------
    img : numpy array
        Image for which the metrics should be calculated.
    brainmask : numpy array
        Brainmask for the image. If provided, the metric will be calculated
        only on the masked region.
    sigma : float
        Standard deviation of the Gaussian filter used
        during canny edge detection.
    n_levels : int
        Levels of intensities to bin image by
    bin : bool
        Whether to bin the image
    crop : bool
        Whether to crop image/ delete empty slices
    weigt_avg : bool
        Whether to calculate the weighted average (depending on the
        proportion of non-zero pixels in the slice).
    reduction : str
        Method to reduce the edge strength values.
        'mean' or 'worst'

    Returns
    -------
    AES : float
        Average Edge Strength measure of the input image.
    """

    #Crop image if crop is True (and no brainmask provided)
    if crop:
        if brainmask is None:
            img = crop_img(img)

    #Bin image if bin is True
    if bin:
        img = bin_img(img, n_levels = n_levels)

    #Centered Gradient kernel in the y- and x-direction
    y_kern = np.array([[-1,-1,-1],
                       [0,0,0],
                       [1,1,1]])
    x_kern = y_kern.T

    #Empty array to contain edge strenghts
    #Function returns the mean of this list
    es = []

    #weights for each slice (proportion of non zero pixels)
    weights = []

    img = img.astype(float)

    for sl in range(img.shape[0]):
        #Slice to do operations on
        im_slice = img[sl]

        #Weight, proportion of non zero pixels
        weights.append(np.mean(im_slice>0))

        #Convolve slice
        x_conv = convolve(im_slice, x_kern)
        y_conv = convolve(im_slice, y_kern)

        #Canny edge detector
        canny_img = canny(im_slice, sigma = sigma)

        if brainmask is not None:
            canny_img = np.ma.masked_array(canny_img,
                                           mask=(brainmask[sl] != 1))
            x_conv = np.ma.masked_array(x_conv,
                                        mask=(brainmask[sl] != 1))
            y_conv = np.ma.masked_array(y_conv,
                                        mask=(brainmask[sl] != 1))

        #Numerator and denominator, to be divided
        #defining the edge strength of the slice
        numerator = np.sum(canny_img*( x_conv**2 + y_conv**2 ))
        denominator = np.sum(canny_img)

        #Append the edge strength
        es.append(np.sqrt(numerator)/denominator)

    es = np.array(es)
    #Remove nans
    es  = es[~np.isnan(es)]

    if reduction == 'worst':
        if weigt_avg:
            raise ValueError("Weighted average not supported for worst reduction")
        return np.min(es)
    elif reduction == 'mean':
        if weigt_avg:
            return np.average(es, weights = weights)
        else:
            return np.mean(es)
    else:
        raise ValueError(f"Reduction method {reduction} not supported.")
