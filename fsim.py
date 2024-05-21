import numpy as np
from image_similarity_measures.quality_metrics import fsim


def calc_masked_FSIM(img, img_ref, mask=None):
    """Calculate FSIM between two 3D images in a specified mask slice-wise.

    Notes:
        - The mask is multiplied to the images and not the FSIM values.
        - Slice dimension is assumed to be the first dimension.
        - The images are assumed to be normalised to [0, 255].
    """

    if mask is not None:
        img *= mask
        img_ref *= mask
        
    fsims = []
    for i in range(len(img)):
        fsims.append(fsim(org_img=img_ref[i][:,  :, None],
                          pred_img=img[i][:,  :, None]))
    return np.array(fsims)
