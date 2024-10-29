import piq
import torch
import numpy as np
from scipy.stats import entropy


def vif(img, img_ref, reduction='mean'):
    """ Calculate the visual information fidelity metric between two images.

    The code is based on the article:
    Sheikh H. R. et al. Image information and visual quality. IEEE
    Transactions on Image Processing. 2006;15(2):430–444.

    Notes:
    - Assume that the images are normalised to [0, 1].
    - The Gaussian noise strength parameter (sigma_n_sq) is set to 0.4,
    according to https://ieeexplore.ieee.org/document/8839547/.

    Parameters
    ----------
    img : numpy array
        image for which the metrics should be calculated.
    img_ref : numpy array
        reference image.
    reduction : str
        Reduction method for the VIF values of multiple slices. Options:
        'mean' (default), 'worst'.

    Returns
    -------
    vif : float
        VIF metric between the two images.
    """

    # add channel dimension
    img = torch.from_numpy(img[:, None]).float()
    img_ref = torch.from_numpy(img_ref[:, None]).float()

    vif_values = piq.vif_p(img, img_ref, data_range=1., sigma_n_sq=2,
                           reduction='none')

    if reduction == 'mean':
        return torch.mean(vif_values).item()
    elif reduction == 'worst':
        return torch.min(vif_values).item()
    else:
        raise ValueError(f"Reduction method {reduction} not supported.")


def image_entropy(img, brainmask=None, reduction='mean'):
    """
    Calculate entropy focus criterion of an image.

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

    ie_slices = []
    for sl in range(img.shape[0]):
        if brainmask is not None:
            img_slice = img[sl][brainmask[sl] == 1]
        else:
            img_slice = img[sl].flatten()

        norm_intensity = img_slice / np.sqrt(np.sum(img_slice ** 2))
        ie_slices.append(-np.nansum(norm_intensity * np.log(norm_intensity)))

    if reduction == 'mean':
        return np.mean(ie_slices)
    elif reduction == 'worst':
        return np.max(ie_slices)
    else:
        raise ValueError(f"Reduction method {reduction} not supported.")
