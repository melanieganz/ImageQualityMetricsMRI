import glob
import numpy as np
import nibabel as nib


def load_data(subject_folder, acq, rec, run):
    """ Load a specific acquisition of a subject."""

    filename = glob.glob(
        "{}/**acq-{}_rec-{}_run-{}**.nii".format(subject_folder, acq,
                                                 rec, run)
    )[0]

    return nib.load(filename).get_fdata().astype(np.uint16)
