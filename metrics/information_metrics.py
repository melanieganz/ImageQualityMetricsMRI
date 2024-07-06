import piq
import torch


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
    # print(img.shape)
    img_ref = torch.from_numpy(img_ref[:, None]).float()
    
    # Function to extend the last dimension to 41 by zero filling
    def extend_last_dim(img):
        zero_padding = torch.zeros(img.shape[0], img.shape[1], 
                                   img.shape[2], 41-img.shape[3])
        img = torch.cat((img, zero_padding), dim=3)
        return img
    
    # Extend the last dimension to 41 if necessary
    # FIXME: this is the minimum number for VIF to run, but not sure why...
    if img.shape[3] < 41:
        img = extend_last_dim(img)
        img_ref = extend_last_dim(img_ref)
    
    vif_values = piq.vif_p(img, img_ref, data_range=1., sigma_n_sq=0.4,
                           reduction='none')

    if reduction == 'mean':
        return torch.mean(vif_values).item()
    elif reduction == 'worst':
        return torch.min(vif_values).item()
    else:
        raise ValueError(f"Reduction method {reduction} not supported.")
