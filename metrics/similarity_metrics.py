import piq
import torch
import numpy as np
from skimage.metrics import structural_similarity


def fsim(img, img_ref, reduction='mean'):
    """Calculate FSIM between two 3D images slice-wise.

    Notes:
        - Slice dimension is assumed to be the first dimension.
        - The images are assumed to be masked (by multiplication)
          and normalised to [0, 1].
        - The final metric value is calculated as mean or min over all slices.
    """

    img = torch.from_numpy(img[:, None]).float()
    img_ref = torch.from_numpy(img_ref[:, None]).float()
    fsim_values = piq.fsim(img, img_ref, data_range=1.0, reduction='none',
                           chromatic=False)

    if reduction == 'mean':
        return torch.mean(fsim_values).item()
    elif reduction == 'worst':
        return torch.min(fsim_values).item()
    else:
        raise ValueError(f"Reduction method {reduction} not supported.")


def ssim(img, img_ref, reduction='mean', brainmask=None):
    """Calculate SSIM between two 3D images slice-wise.

    Notes:
        - Slice dimension is assumed to be the first dimension.
        - The images are assumed to be normalised to [0, 1].
        - If brainmask is not None, the calculated (full) SSIM values are
        masked with the brainmask.
        - The final metric value is calculated as mean or min over all slices.
    """

    _, ssim_values = structural_similarity(img, img_ref, data_range=1.0,
                                           full=True)
    if brainmask is not None:
        masked_ssim = np.ma.masked_array(ssim_values, mask=(brainmask != 1))
    ssim_values = np.mean(ssim_values, axis=(1, 2))

    if reduction == 'mean':
        return np.mean(ssim_values)
    elif reduction == 'worst':
        return np.min(ssim_values)
    else:
        raise ValueError(f"Reduction method {reduction} not supported.")
