import warnings
warnings.filterwarnings("ignore")
import os
import argparse
from data_utils import *
from metrics.similarity_metrics import fsim, ssim, psnr
from metrics.perceptual_metrics import lpips
from metrics.information_metrics import vif, image_entropy
from metrics.gradient_metrics import *
from archive.CoEnt import *


def compute_metrics(filename, subject, acquisition, output_file, brainmask_file="none",
                    ref_file=False, normal="min_max", mask_metric_values=False,
                    reduction='worst'):
    """
    Calculate metrics for a given image.

    Parameters
    ----------
    filename : str
        filename of the nifti image which is supposed to be evaluated.
    subject : str
        subject identifier.
    output_file : str
        filename for the output csv file.
    brainmask_file : str, optional.
        filename for the corresponding brainmask. If it is set to None, the
        metric will be calculated on the whole image.The default is None.
    ref_file : str, optional
        filename for the reference nifti scan which the image is supposed to
        be compared to. This is only needed for SSIM and PSNR. The default is
        False.
    normal : str, optional
        whether and how the data should be normalized before metric calculation.
        The default is min_max normalisation. Other options: "mean_std" or "none".
        Note: mean_std normalisation is not applicable to all metrics.
    mask_metric_values : bool, optional
        whether the brainmask should be multiplied to the images (False) or
        used to mask the metric values (True). Only applicable to some metrics.
    reduction : str, optional
        reduction method for the metric calculation. The default is 'worst'.

    Returns
    -------
    res : float
        value of the metric.
        :param subject:

    """
    
    metrics_dict = {
        "full_reference": {
            'SSIM':ssim,
            'PSNR':psnr,
            "FSIM": fsim,
            "VIF": vif,
            "LPIPS": lpips},

        "reference_free": {
            "AES": aes,
            "TG": tenengrad,
            "NGS": normalized_gradient_squared,
            "GE": gradient_entropy,
            "IE": image_entropy
            }
    }

    # Load data, brainmask and reference:
    img = nib.load(filename).get_fdata().astype(np.uint16)
    
    if brainmask_file != "none":
        brainmask = nib.load(brainmask_file).get_fdata().astype(np.uint16)
    else:
        brainmask = None
        
    if ref_file:
        ref = nib.load(ref_file).get_fdata().astype(np.uint16)
    else:
        ref = None

    # Roll the z-axis to the first dimension if the sequence is t1tirm or t2tse
    if "t1tirm" in filename or "t2tse" in filename:
        img = np.rollaxis(img, 2)
        ref = (np.rollaxis(ref, 2) if ref is not None else None)
        brainmask = (np.rollaxis(brainmask, 2) if brainmask is not None
                     else None)

    # Apply brainmask if available:
    if brainmask is not None:
        img_masked = np.multiply(img, brainmask)
        ref_masked = (np.multiply(ref, brainmask) if ref is not None else None)
    else:
        img_masked = img
        ref_masked = ref

    # Sort out the slices with only zeros:
    img_masked, ref_masked, brainmask = sort_out_zero_slices(img_masked,
                                                             ref_masked,
                                                             brainmask)

    # Normalization:
    if normal == "min_max":
        img = min_max_scale(img_masked)
        ref = (min_max_scale(ref_masked) if ref is not None else None)
    elif normal == "mean_std":
        img = normalize_mean_std(img_masked)
        ref = (normalize_mean_std(ref_masked) if ref is not None else None)
        metrics_dict["full_reference"].pop("FSIM")
        metrics_dict["full_reference"].pop("VIF")
        metrics_dict["full_reference"].pop("LPIPS")
    elif normal == "none":
        img = img_masked
        ref = ref_masked
        metrics_dict["full_reference"].pop("FSIM")
        metrics_dict["full_reference"].pop("VIF")
        metrics_dict["full_reference"].pop("LPIPS")
    elif normal == "percentile":
        img = normalize_percentile(img_masked)
        ref = normalize_percentile(ref_masked)
    else:
        raise ValueError("Normalisation method not recognized.")
    
    res = []
    for m in metrics_dict["full_reference"]:
        if m in ["SSIM", "PSNR"]:
            if mask_metric_values:
                metric_value = metrics_dict['full_reference'][m](
                    img, ref, reduction=reduction, brainmask=brainmask
                )
            else:
                metric_value = metrics_dict['full_reference'][m](
                    img, ref, reduction=reduction, brainmask=None
                )
        else:
            metric_value = metrics_dict['full_reference'][m](
                img, ref, reduction=reduction
            )

        print(f"{m}: {metric_value}")
        res = np.append(res,metric_value)

    for m in metrics_dict["reference_free"]:
        if mask_metric_values:
            metric_value = metrics_dict['reference_free'][m](
                img, brainmask, reduction=reduction
            )
        else:
            metric_value = metrics_dict['reference_free'][m](
                img, reduction=reduction
            )

        print(f"{m}: {metric_value}")
        res = np.append(res,metric_value)

    if not os.path.exists(output_file):
        with open(output_file, 'a') as f:
            f.write("Sbj,Acq,File,"
                    + ",".join( metrics_dict["full_reference"].keys())
                    + "," + ",".join(metrics_dict["reference_free"].keys())
                    + "\n"
                    )

    with open(output_file, 'a') as f:
        f.write(f"{subject},{acquisition},{os.path.basename(filename)}," + ",".join(map(str, res)) + "\n")

    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute image quality metrics.")
    parser.add_argument("filename", type=str,
                        help="Filename of the nifti image to be evaluated.")
    parser.add_argument("subject", type=str,
                        help="Subject identifier.")
    parser.add_argument("acquisition", type=str,
                        help="Acquisition identifier.")
    parser.add_argument("output_file", type=str,
                        help="Filename for the output CSV file.")
    parser.add_argument("brainmask_file", type=str, nargs='?',
                        default=False,
                        help="Filename for the corresponding brainmask (optional).")
    parser.add_argument("ref_file", type=str, nargs='?',
                        default=False,
                        help="Filename for the reference nifti scan (optional).")
    parser.add_argument("--normal", type=str, default=True,
                        help="Whether to normalize the data before metric "
                             "calculation (default: True).")
    parser.add_argument("--mask_metric_values", type=str,
                        default="False",
                        help="Whether to use the brainmask to mask the metric "
                             "values (default: False).")
    parser.add_argument("--reduction", type=str, default='worst',
                        help="Reduction method for the metric calculation "
                             "(default: 'worst').")
    args = parser.parse_args()
    if args.mask_metric_values == "True":
        mask_metric_values = True
    elif args.mask_metric_values == "False":
        mask_metric_values = False
    else:
        raise ValueError("mask_metric_values must be either 'True' or 'False'.")

    compute_metrics(args.filename, args.subject, args.acquisition, args.output_file,
                    args.brainmask_file, args.ref_file, args.normal,
                    mask_metric_values, args.reduction)
