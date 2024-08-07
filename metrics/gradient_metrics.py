import numpy as np
from scipy.ndimage import sobel
from scipy.stats import entropy
from skimage.feature import canny
from scipy.ndimage import convolve
from data_utils import crop_img, bin_img


def calc_gradient_magnitude(img, mode="2d"):
    """Calculate the magnitude of the image gradient.

    Note:
        - The image is assumed to be a 3D image.
        - The image is assumed to be masked and normalised to [0, 1].
        - The image is converted to floating point numbers for a correct
        calculation of the gradient.
    """

    img = img.astype(float)

    grad_x = sobel(img, axis=1, mode='reflect')
    grad_y = sobel(img, axis=2, mode='reflect')

    if mode == "2d":
        return np.sqrt(grad_x ** 2 + grad_y ** 2)
    elif mode == "3d":
        grad_z = sobel(img, axis=0, mode='reflect')
        return np.sqrt(grad_x ** 2 + grad_y ** 2 + grad_z ** 2)
    else:
        raise ValueError(f"Mode {mode} not supported.")


def tenengrad(img, brainmask=None, reduction='mean'):
    """Tenengrad measure of the input image.

    The code is based on the article:
    Krotkov E. Focusing. Int J Comput Vis. 1988; 1(3):223-237

    Parameters
    ----------
    img : numpy array
        image for which the metrics should be calculated.
    brainmask : boolean True or False, optional
        If True, a brainmask was used to mask the images before 
        calculating the metrics. Image is flattened prior metric 
        estimation. The default is False.

    Returns
    -------
    tg : float
        Tenengrad measure of the input image.
    """

    grad = calc_gradient_magnitude(img, mode="2d")

    if brainmask is not None:
        grad = np.ma.masked_array(grad, mask=(brainmask != 1))

    if reduction == 'mean':
        return np.mean(grad ** 2)
    elif reduction == 'worst':
        grad_slices = np.mean(grad ** 2, axis=(1, 2))
        return np.min(grad_slices)
    else:
        raise ValueError(f"Reduction method {reduction} not supported.")


def gradient_entropy(img, brainmask=None, reduction='mean'):
    """
    Calculate gradient entropy of an image.

    Reference:
    Atkinson, D.; Hill, D. L.; Stoyle, P. N.; Summers, P. E.; Keevil, S. F.
    (1997): Automatic correction of motion artifacts in magnetic resonance
    images using an entropy focus criterion. In IEEE Transactions on Medical
    Imaging 16 (6), pp. 903–910. DOI: 10.1109/42.650886.

    Parameters
    ----------
    img : numpy array
        image for which the metrics should be calculated.
    brainmask : bool, optional
        Whether the metric values should be masked with the brainmask. If None,
        no masking is performed.
    reduction : str, optional
        Reduction method for the image entropy values of multiple slices.
        Options: 'mean' (default), 'worst'.

    Returns
    -------
    ie : float
        Image Entropy of the input image.
    """

    grad = calc_gradient_magnitude(img, mode="2d")

    ge_slices = []
    for sl in range(img.shape[0]):
        if brainmask is not None:
            grad_slice = grad[sl][brainmask[sl] == 1]
        else:
            grad_slice = grad[sl].flatten()

        norm_intensity = grad_slice / np.sqrt(np.sum(grad_slice ** 2))
        ge_slices.append(-np.nansum(norm_intensity * np.log(norm_intensity)))

    if reduction == 'mean':
        return np.mean(ge_slices)
    elif reduction == 'worst':
        return np.max(ge_slices)
    else:
        raise ValueError(f"Reduction method {reduction} not supported.")


def gradient_entropy_depr(img, brainmask=None, reduction='mean'):
    """Gradient Entropy of the input image.

    DEPRECATED. Uses counts and not the actual image intensities, which
    does not make sense in this application.

    The code is based on the article:
    McGee K, Manduca A, Felmlee J et al. Image metric-based correction
    (autocorrection) of motion effects: analysis of image metrics. J Magn Reson
    Imaging. 2000; 11(2):174-181.

    Parameters
    ----------
    img : numpy array
        image for which the metrics should be calculated.
    brainmask : boolean True or False, optional
        If True, a brainmask was used to mask the images before 
        calculating the metrics. Image is flattened prior metric 
        estimation. The default is False.

    Returns
    -------
    ge : float
        Gradient Entropy of the input image.
    """

    grad = calc_gradient_magnitude(img, mode="2d")  # maybe needs to be normalized

    ge_slices = []
    for sl in range(img.shape[0]):
        if brainmask is not None:
            grad_slice = grad[sl][brainmask[sl] == 1]
        else:
            grad_slice = grad[sl].flatten()
        _, counts = np.unique(grad_slice, return_counts=True)
        ge_slices.append(entropy(counts, base=2))

    if reduction == 'mean':
        return np.mean(ge_slices)
    elif reduction == 'worst':
        return np.max(ge_slices)
    else:
        raise ValueError(f"Reduction method {reduction} not supported.")


def normalized_gradient_squared(img, brainmask=None, reduction='mean'):
    """Normalized gradient squared measure of the input image.

    The code is based on the article:
    McGee K, Manduca A, Felmlee J et al. Image metric-based correction
    (autocorrection) of motion effects: analysis of image metrics. J Magn Reson
    Imaging. 2000; 11(2):174-181.

    Note:
    - The value of the metric is scaled by the number of pixels in the image
       to avoid the value being too small.

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
    ngs : float
        Normalized gradient squared measure of the input image.
    """

    grad = calc_gradient_magnitude(img, mode="2d")

    if brainmask is not None:
        grad = np.ma.masked_array(grad, mask=(brainmask != 1))

    ngs_values = np.sum((grad / np.sum(grad)) ** 2, axis=(1, 2)) * grad[0].size

    if reduction == 'mean':
        return np.mean(ngs_values)
    elif reduction == 'worst':
        return np.min(ngs_values)
    else:
        raise ValueError(f"Reduction method {reduction} not supported.")


def aes(img, brainmask=None, sigma=np.sqrt(2), n_levels=128, bin=False,
        crop=True, weigt_avg=False, reduction='mean'):
    """
    Calculate the metric Average Edge Strength.
    Original code my Simon Chemnitz-Thomsen.

    Reference:
    Quantitative framework for prospective motion correction evaluation
    Nicolas Pannetier, Theano Stavrinos, Peter Ng, Michael Herbst,
    Maxim Zaitsev, Karl Young, Gerald Matson, and Norbert Schuff

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

    # Crop image if crop is True (and no brainmask provided)
    if crop:
        if brainmask is None:
            img = crop_img(img)

    # Bin image if bin is True
    if bin:
        img = bin_img(img, n_levels=n_levels)

    # Centered Gradient kernel in the y- and x-direction
    y_kern = np.array([[-1, -1, -1],
                       [0, 0, 0],
                       [1, 1, 1]])
    x_kern = y_kern.T

    # Empty array to contain edge strenghts
    # Function returns the mean of this list
    es = []

    # weights for each slice (proportion of non zero pixels)
    weights = []

    img = img.astype(float)

    for sl in range(img.shape[0]):
        # Slice to do operations on
        im_slice = img[sl]

        # Weight, proportion of non zero pixels
        weights.append(np.mean(im_slice > 0))

        # Convolve slice
        x_conv = convolve(im_slice, x_kern)
        y_conv = convolve(im_slice, y_kern)

        # Canny edge detector
        canny_img = canny(im_slice, sigma=sigma)

        if brainmask is not None:
            canny_img = np.ma.masked_array(canny_img,
                                           mask=(brainmask[sl] != 1))
            x_conv = np.ma.masked_array(x_conv,
                                        mask=(brainmask[sl] != 1))
            y_conv = np.ma.masked_array(y_conv,
                                        mask=(brainmask[sl] != 1))

        # Numerator and denominator, to be divided
        # defining the edge strength of the slice
        numerator = np.sum(canny_img * (x_conv ** 2 + y_conv ** 2))
        denominator = np.sum(canny_img)

        # Append the edge strength
        es.append(np.sqrt(numerator) / denominator)

    es = np.array(es)
    # Remove nans
    es = es[~np.isnan(es)]

    if reduction == 'worst':
        if weigt_avg:
            raise ValueError(
                "Weighted average not supported for worst reduction")
        return np.min(es)
    elif reduction == 'mean':
        if weigt_avg:
            return np.average(es, weights=weights)
        else:
            return np.mean(es)
    else:
        raise ValueError(f"Reduction method {reduction} not supported.")
