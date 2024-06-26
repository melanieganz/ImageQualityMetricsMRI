import numpy as np
import image_similarity_measures as img_sim


def fsim(img, img_ref):
    """Calculate FSIM between two 3D images slice-wise.

    Notes:
        - Slice dimension is assumed to be the first dimension.
        - The images are assumed to be masked and normalised to [0, 255].
    """
        
    fsims = []
    for i in range(len(img)):
        fsims.append(img_sim.quality_metrics.fsim(
            org_img=img_ref[i][:,  :, None],
            pred_img=img[i][:,  :, None])
        )

    return np.mean(fsims)
