import matplotlib.pyplot as plt
from utils.data_utils import *


def plot_histograms(img1, img2, label1, label2, mask=None, xlim=(0,1500),
                    ylim=0.5e6, title=""):

    if mask is not None:
        img1 = img1[mask > 0]
        img2 = img2[mask > 0]
    bins = np.linspace(xlim[0], xlim[1], 71)

    plt.figure(figsize=(8, 6))
    plt.hist(img1.flatten(), bins=bins, color="tab:blue", histtype='step',
             label=label1, linewidth=2)
    plt.hist(img2.flatten(), bins=bins, color="tab:red", histtype='step',
                label=label2, linewidth=2)
    plt.xlim(xlim[0], xlim[1])
    plt.ylim(0, ylim)
    plt.legend(loc='best', fontsize=25)
    plt.ylabel("Counts", fontsize=25)
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    plt.tick_params(axis='y', which='both', labelsize=21)
    plt.gca().yaxis.get_offset_text().set_fontsize(21)
    plt.xlabel("Pixel intensity", fontsize=25)
    plt.xticks(fontsize=21)
    plt.locator_params(axis='x', nbins=6)
    plt.grid(axis='y', color='lightgrey', linestyle='-', linewidth=0.5)
    plt.title(title, fontsize=25)
    plt.tight_layout()
    plt.show()


def show_example_slice(img, ref, slice_idx=50):
    """
    Display an example slice of the image and its reference using plt.imshow.

    Parameters:
    img (ndarray): The image data.
    ref (ndarray): The reference data.
    slice_idx (int, optional): The index of the slice to display. Default is 0.
    """

    # Calculate vmin and vmax
    vmin = min(np.min(img), np.min(ref))
    vmax = max(np.max(img), np.max(ref))

    # Display the image slice
    plt.figure(figsize=(6, 6*206/256))
    plt.imshow(img[slice_idx, :, 50:][::-1, ::-1].T, cmap='gray', vmin=vmin, vmax=vmax)
    # plt.title('Image')
    plt.text(0.5, 0.96, 'Image', color='white', fontsize=40, ha='center',
             va='top', transform=plt.gca().transAxes)
    plt.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.show()

    # Display the reference slice
    plt.figure(figsize=(6, 6*206/256))
    plt.imshow(ref[slice_idx, :, 50:][::-1, ::-1].T, cmap='gray', vmin=vmin, vmax=vmax)
    # plt.title('Reference')
    plt.text(0.5, 0.96, 'Reference', color='white', fontsize=40, ha='center',
             va='top', transform=plt.gca().transAxes)
    plt.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
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

    ylim = 0.12e6
    # no normalization - within brainmask:
    plot_histograms(img_masked, ref_masked, "Image",
                    "Reference", mask=brainmask, #title="None",
                    ylim=ylim, xlim=(0, 800))
    show_example_slice(img_masked, ref_masked, slice_idx=50)

    # min-max normalisation
    img_norm = min_max_scale(img_masked)
    ref_norm = min_max_scale(ref_masked)
    plot_histograms(img_norm, ref_norm, "Image", "Reference",
                    mask=brainmask,
                    # title="Min-max",
                    xlim=(0,1), ylim=ylim)
    show_example_slice(img_norm, ref_norm, slice_idx=50)

    # mean-std normalisation
    img_norm = normalize_mean_std(img_masked)
    ref_norm = normalize_mean_std(ref_masked)
    plot_histograms(img_norm, ref_norm, "Image", "Reference",
                    mask=brainmask,
                    # title="Mean-std",
                    xlim=(-1,4), ylim=ylim)
    show_example_slice(img_norm, ref_norm, slice_idx=50)

    # percentile normalization
    img_norm = normalize_percentile(img_masked)
    ref_norm = normalize_percentile(ref_masked)
    plot_histograms(img_norm, ref_norm, "Image", "Reference",
                    mask=brainmask,
                    # title="Percentile",
                    xlim=(-0.5,1.5), ylim=ylim)
    show_example_slice(img_norm, ref_norm, slice_idx=50)


compare_histograms(filename='OpenNeuro_dataset/sub-01/mprage/align_sub'
                            '-01_acq-mpragepmcon_rec-wre_run-02_T1w.nii.gz',
                   brainmask_file='OpenNeuro_dataset/sub-01/mprage/'
                                  'bet_mprage_mask.nii.gz',
                   ref_file='OpenNeuro_dataset/sub-01/mprage/'
                            'ref_mprage_image.nii')

