import nibabel as nib
import numpy as np


def load_nifti_file(file_path: str) -> np.ndarray:
    nifti = nib.load(file_path)
    return nifti.get_fdata()


def get_standardized_slice(array: np.array, slice_number: int) -> np.array:
    array = array[:, :, slice_number]
    array = array - np.min(array)
    divider = np.max(array) if np.max(array) > 0 else 1
    array = array * 255 / divider
    return array


def save_as_nifti(array: np.array, nifti_path, affine=np.eye(4)) -> None:
    nifti_array = nib.Nifti1Image(array, affine)
    nib.save(nifti_array, nifti_path)
