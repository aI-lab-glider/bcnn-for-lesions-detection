import nibabel as nib
import numpy as np


def get_patient_index(mask_path: str) -> str:
    """
    Extracts patient index from the path to mask.
    :param mask_path: path to patient mask
    :return: patient index
    """
    return mask_path.split('.')[0].split('_')[-1]


def load_nifti_file(file_path: str) -> np.ndarray:
    nifti = nib.load(file_path)
    return nifti.get_fdata()


def save_as_nifti(array: np.array, nifti_path, affine=np.eye(4)) -> None:
    nifti_array = nib.Nifti1Image(array, affine)
    nib.save(nifti_array, nifti_path)
