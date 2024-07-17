import numpy as np
from scipy.ndimage import sobel
from scipy.stats import entropy


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
    """Gradient Entropy of the input image.

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


def normalized_gradient_squared(img, brainmask=None):
    """Normalized gradient squared measure of the input image.

    The code is based on the article:
    McGee K, Manduca A, Felmlee J et al. Image metric-based correction
    (autocorrection) of motion effects: analysis of image metrics. J Magn Reson
    Imaging. 2000; 11(2):174-181.

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

    grad = calc_gradient_magnitude(img)

    # apply brainmask:
    if brainmask is not None:
        grad = grad.flatten()
        grad = grad[grad > 0]

    #FIXME: added a scaling factor here otherwise the value is too small, not sure if it is a good idea? 
    return np.sum((grad / np.sum(grad)) ** 2) * len(grad)
