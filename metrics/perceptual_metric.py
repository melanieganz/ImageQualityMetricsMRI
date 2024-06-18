import lpips
import torch
import numpy as np


def perceptual_metric(img, img_ref):
    """ Calculate the perceptual metric between two images.

    Notes:
    - Assume that the images are normalised to [0, 1].
    - The images are assumed to be 3D images and processed slice-wise,
    with the final score being calculated as mean.

    Parameters
    ----------
    img : numpy array
        image for which the metrics should be calculated.
    img_ref : numpy array
        reference image.

    Returns
    -------
    loss_vgg : float
        perceptual metric (lpips) between the two images.
    """

    loss_fn_vgg = lpips.LPIPS(net='vgg')

    img = torch.from_numpy(img[:, None]).float() * 2 - 1
    img_ref = torch.from_numpy(img_ref[:, None]).float() * 2 - 1

    loss_vgg =  loss_fn_vgg(img, img_ref).detach().numpy()

    return np.mean(loss_vgg)
