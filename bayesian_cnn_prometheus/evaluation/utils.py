import json
from pathlib import Path
import nibabel as nib
import numpy as np

from bayesian_cnn_prometheus.constants import Paths


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


def get_patient_index(mask_path: str) -> str:
    """
    Extracts patient index from the path to mask.
    :param mask_path: path to patient mask
    :return: patient index
    """
    return mask_path.split('.')[0].split('_')[-1]


def load_config(path: Path = Paths.CONFIG_PATH):
    with open(path) as cf:
        config = json.load(cf)
    return config


def assert_fields_have_values(values_as_dict, required_keys=None):
    """
    Asserts, that all :param required_keys: have assigned values in :param values_as_dict:.
    In case if no :param required_keys: provided, asserts, that all fields have assigned values.
    """
    required_keys = required_keys or values_as_dict.keys()
    unassigned_keys = [
        key for key in required_keys if values_as_dict.get(key, None) is None]
    assert not unassigned_keys, f"no values provided for required keys: {', '.join(unassigned_keys)}"
