import numpy as np
from image_similarity_measures.quality_metrics import fsim


def calc_fsim(img, img_ref):
    """Calculate FSIM between two 3D images slice-wise.

    Notes:
        - Slice dimension is assumed to be the first dimension.
        - The images are assumed to be masked and normalised to [0, 255].
    """
        
    fsims = []
    for i in range(len(img)):
        fsims.append(fsim(org_img=img_ref[i][:,  :, None],
                          pred_img=img[i][:,  :, None]))

    return np.mean(fsims)
