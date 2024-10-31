from lpips import LPIPS
import torch
import numpy as np


def lpips(img, img_ref, reduction='mean'):
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
    reduction : str
        reduction method for the LPIPS values of multiple slices. Options:
        'mean' (default), 'worst' which returns the maximum LPIPS value.

    Returns
    -------
    loss_vgg : float
        perceptual metric (lpips) between the two images.
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loss_fn_vgg = LPIPS(net='vgg', verbose=False).to(device)

    # change range of images to [-1, 1] and add RGB channel dimension
    img = torch.from_numpy(img[:, None]).float() * 2 - 1
    img_ref = torch.from_numpy(img_ref[:, None]).float() * 2 - 1
    img = img.repeat(1, 3, 1, 1).to(device)
    img_ref = img_ref.repeat(1, 3, 1, 1).to(device)

    loss_vgg =  loss_fn_vgg(img, img_ref)

    if reduction == 'mean':
        return torch.mean(loss_vgg).item()
    elif reduction == 'worst':
        return torch.max(loss_vgg).item()
    else:
        raise ValueError(f"Reduction method {reduction} not supported.")
