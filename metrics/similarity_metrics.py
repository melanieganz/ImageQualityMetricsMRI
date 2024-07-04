import piq
import torch


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
    elif reduction == 'min':
        return torch.min(fsim_values).item()
    else:
        raise ValueError(f"Reduction method {reduction} not supported.")
