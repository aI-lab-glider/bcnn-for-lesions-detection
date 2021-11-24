from pathlib import Path
from typing import Tuple

import nibabel as nib
import numpy as np

from bayesian_cnn_prometheus.constants import Paths


# EXP: If it loads, it's not a transformator!
class ImageLoader:

    def __init__(self, ext: str = 'nii.gz'):
        self.extension = ext

    def load(self, image_index: str) -> Tuple[np.array, np.array]:
        """
        Transforms image and target with image_index from nifti format into numpy array.
        :param image_index: index of image to be transformed
        :return: image and target as numpy arrays
        """
        image_file_path, target_file_path = self._get_files_names(image_index, 'nii.gz')
        if self.extension == 'nii.gz':
            npy_image = self._transform_nifti_to_npy(image_file_path)
            npy_target = self._transform_nifti_to_npy(target_file_path)
        elif self.extension == 'npy':
            npy_image = np.load(str(image_file_path))
            npy_target = np.load(str(target_file_path))
        else:
            raise Exception("Not supported extension")
        return npy_image, npy_target

    @staticmethod
    def _transform_nifti_to_npy(nifti_file_path: Path) -> np.array:
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
        image_file_path = str(Paths.IMAGE_FILE_PATTERN_PATH).format(image_index, file_format)
        target_file_path = str(Paths.REFERENCE_SEGMENTATION_FILE_PATTERN_PATH).format(image_index, file_format)
        return Path(image_file_path), Path(target_file_path)
