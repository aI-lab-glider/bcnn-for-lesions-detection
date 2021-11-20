import nibabel as nib
import numpy as np


def load_nifti_file(file_path: str) -> np.ndarray:
    nifti = nib.load(file_path)
    return nifti.get_fdata()