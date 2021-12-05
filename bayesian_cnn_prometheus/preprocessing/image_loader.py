from pathlib import Path
from typing import Tuple
from bayesian_cnn_prometheus.evaluation.utils import standardize_image

import nibabel as nib
import numpy as np

from bayesian_cnn_prometheus.constants import Paths


class ImageLoader:

    def __init__(self, ext: str = 'nii.gz'):
        self.extension = ext

    def load(self, image_index: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transforms image and target with image_index from nifti format into numpy array.
        :param image_index: index of image to be transformed
        :return: image and target as numpy arrays
        """
        image_file_path, target_file_path = self._get_files_names(
            image_index, 'nii.gz')
        target = self._load_image(target_file_path)
        image = self._load_image(image_file_path)
        return image, self._unify_segmentation_labels(target)

    @staticmethod
    def _unify_segmentation_labels(image):
        image[image == 1] = 0  # to remove trachea
        return image.astype(bool).astype('int16')

    def _load_image(self, path):
        if self.extension == 'nii.gz':
            return self._load_nifti_as_npy(path)
        elif self.extension == 'npy':
            return np.load(str(path))
        else:
            raise Exception("Not supported extension")

    @staticmethod
    def _load_nifti_as_npy(nifti_file_path: Path) -> np.ndarray:
        """
        Transforms single nifti file into the numpy array.
        :param nifti_file_path: path to nifti file
        :return: image or target in numpy array format
        """
        if nifti_file_path.exists():
            nifti = nib.load(nifti_file_path)
            npy = nifti.get_fdata()
            return npy
        else:
            raise Exception(f'File {nifti_file_path} does not exist!')

    @staticmethod
    def _get_files_names(image_index: str, file_format: str) -> Tuple[Path, Path]:
        """
        On the base of the image index generates paths to image and target arrays.
        :param image_index: index of the image to be transformed
        :param file_format: format of the file with original image or target
        :return: paths to image and target to be transformed
        """
        image_file_path = str(Paths.IMAGE_FILE_PATTERN_PATH).format(
            f'{image_index:0>4}', file_format)
        target_file_path = str(Paths.REFERENCE_SEGMENTATION_FILE_PATTERN_PATH).format(
            f'{image_index:0>4}', file_format)
        return Path(image_file_path), Path(target_file_path)
