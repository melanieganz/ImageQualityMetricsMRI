import numpy as np
from scipy.ndimage import sobel
from scipy.stats import entropy


def calc_gradient_magnitude(img):
    """Calculate the magnitude of the image gradient.

    Note:
        - The image is assumed to be a 3D image.
        - The image is assumed to be masked and normalised to [0, 1].
        - The image is converted to floating point numbers for a correct
        calculation of the gradient.
    """

    img = img.astype(float)

    grad_x = sobel(img, axis=0, mode='reflect')
    grad_y = sobel(img, axis=1, mode='reflect')
    grad_z = sobel(img, axis=2, mode='reflect')

    return np.sqrt(grad_x ** 2 + grad_y ** 2 + grad_z ** 2)


def tenengrad(img, brainmask=None):
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

    grad = calc_gradient_magnitude(img)

    # apply flattened brainmask:
    if brainmask is not None:
        grad = grad.flatten()
        grad = grad[grad > 0]

    return np.mean(grad ** 2)


def gradient_entropy(img, brainmask=None):
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

    grad = calc_gradient_magnitude(img)  # maybe needs to be normalized

    # apply flattened brainmask:
    if brainmask is not None:
        grad = grad.flatten()
        grad = grad[grad > 0]

    _, counts = np.unique(grad, return_counts=True)
    ge = entropy(counts, base=2)

    return ge


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
