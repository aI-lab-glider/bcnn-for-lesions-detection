from pathlib import Path
import nibabel as nib
import numpy as np


def load_nifti_file(file_path: str) -> np.ndarray:
    nifti = nib.load(file_path)
    return nifti.get_fdata()


def standardize_image(array) -> np.ndarray:
    array = array - np.min(array)
    divider = np.max(array) if np.max(array) != 0 else 1
    array = array * 255 / divider
    return array.astype(np.int16)


def get_standardized_slice(array: np.ndarray, slice_number: int) -> np.ndarray:
    return standardize_image(array[:, :, slice_number])


def save_as_nifti(array: np.ndarray, nifti_path: Path, affine=np.eye(4), header=None) -> None:
    if not nifti_path.parent.exists():
        nifti_path.parent.mkdir(parents=True)
    nifti_img = nib.Nifti1Image(array, affine, header=header)
    nib.save(nifti_img, nifti_path)
