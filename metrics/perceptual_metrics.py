from lpips import LPIPS
import torch
import numpy as np


def lpips(img, img_ref):
    """ Calculate the perceptual metric between two images.

    The code is based on the article:
    Zhang, R. et al. The Unreasonable Effectiveness of Deep Features as a
    Perceptual Metric. CVPR 2018.

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

    loss_fn_vgg = LPIPS(net='vgg')

    # change range of images to [-1, 1] and add RGB channel dimension
    img = torch.from_numpy(img[:, None]).float() * 2 - 1
    img_ref = torch.from_numpy(img_ref[:, None]).float() * 2 - 1
    img = img.repeat(1, 3, 1, 1)
    img_ref = img_ref.repeat(1, 3, 1, 1)

    loss_vgg =  loss_fn_vgg(img, img_ref).detach().numpy()

    return np.mean(loss_vgg)
