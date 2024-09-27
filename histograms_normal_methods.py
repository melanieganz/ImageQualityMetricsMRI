import matplotlib.pyplot as plt
from sympy.printing.pretty.pretty_symbology import line_width

from data_utils import *


def plot_histograms(img1, img2, label1, label2, xlim=(0,1500), ylim=0.5e6,
                    title=""):
    bins = np.linspace(xlim[0], xlim[1], 71)
    plt.hist(img1.flatten(), bins=bins, color="tab:blue", histtype='step',
             label=label1, linewidth=2)
    plt.hist(img2.flatten(), bins=bins, color="tab:green", histtype='step',
                label=label2, linewidth=2)
    plt.xlim(xlim[0], xlim[1])
    plt.ylim(0, ylim)
    plt.legend(loc='best', fontsize=14)
    plt.ylabel("Counts", fontsize=14)
    plt.xlabel("Pixel intensity", fontsize=14)
    plt.title(title, fontsize=16)
    plt.tight_layout()
    plt.show()


def compare_histograms(filename, brainmask_file="none", ref_file=False):

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

    # no preprocessing:
    plot_histograms(img, ref, "Image", "Reference", title="Original / not masked",
                    ylim=0.3e6)

    # brain masking:
    plot_histograms(img_masked, ref_masked, "Image",
                    "Reference", title="No normalisation", ylim=0.3e6)

    # min-max normalisation
    img_norm = min_max_scale(img_masked)
    ref_norm = min_max_scale(ref_masked)

    plot_histograms(img_norm, ref_norm, "Image", "Reference",
                    title="Min-Max Normalisation", xlim=(0,1), ylim=0.3e6)

    # mean-std normalisation
    img_norm = normalize_mean_std(img_masked)
    ref_norm = normalize_mean_std(ref_masked)

    plot_histograms(img_norm, ref_norm, "Image", "Reference",
                    title="Mean-Std Normalisation", xlim=(0,4), ylim=0.3e6)

    # percentile normalisation no clipping
    img_norm = normalize_percentile(img_masked, clip=False, upper_percentile=98,
                                    lower_percentile=2)
    ref_norm = normalize_percentile(ref_masked, clip=False, upper_percentile=98,
                                    lower_percentile=2)

    plot_histograms(img_norm, ref_norm, "Image", "Reference",
                    title="Percentile Normalisation", xlim=(-1,2), ylim=0.3e6)

    img_norm = normalize_percentile(img_masked, clip=False, upper_percentile=99,
                                    lower_percentile=1)
    ref_norm = normalize_percentile(ref_masked, clip=False, upper_percentile=99,
                                    lower_percentile=1)

    plot_histograms(img_norm, ref_norm, "Image", "Reference",
                    title="Percentile Normalisation", xlim=(-1,2), ylim=0.3e6)

    img_norm = normalize_percentile(img_masked, clip=False, upper_percentile=99.9,
                                    lower_percentile=1)
    ref_norm = normalize_percentile(ref_masked, clip=False, upper_percentile=99.9,
                                    lower_percentile=1)

    plot_histograms(img_norm, ref_norm, "Image", "Reference",
                    title="Percentile Normalisation", xlim=(-1,2), ylim=0.3e6)


    img_norm = normalize_percentile(img_masked, clip=True, upper_percentile=99.9,
                                    lower_percentile=1)
    ref_norm = normalize_percentile(ref_masked, clip=True, upper_percentile=99.9,
                                    lower_percentile=1)

    plot_histograms(img_norm, ref_norm, "Image", "Reference",
                    title="Percentile Normalisation", xlim=(-1,2), ylim=0.3e6)


compare_histograms(filename='OpenNeuro_dataset/sub-01/mprage/align_sub'
                            '-01_acq-mpragepmcoff_rec-wore_run-02_T1w.nii.gz',
                   brainmask_file='OpenNeuro_dataset/sub-01/mprage/'
                                  'bet_mprage_mask.nii.gz',
                   ref_file='OpenNeuro_dataset/sub-01/mprage/'
                            'ref_mprage_image.nii')


compare_histograms(filename='OpenNeuro_dataset/sub-01/mprage/align_sub'
                            '-01_acq-mpragepmcon_rec-wre_run-02_T1w.nii.gz',
                   brainmask_file='OpenNeuro_dataset/sub-01/mprage/'
                                  'bet_mprage_mask.nii.gz',
                   ref_file='OpenNeuro_dataset/sub-01/mprage/'
                            'ref_mprage_image.nii')

