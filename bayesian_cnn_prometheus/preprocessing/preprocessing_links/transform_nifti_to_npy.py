import os
from typing import Dict, Tuple

import nibabel as nib
import numpy as np

from bayesian_cnn_prometheus.constants import DATA_DIR, IMAGES_DIR, REFERENCE_SEGMENTATIONS_DIR
from bayesian_cnn_prometheus.preprocessing.preprocessing_links.chain_link import ChainLink


class TransformNiftiToNpy(ChainLink):
    def run(self, global_config: Dict[str, str], image_index: str) -> Tuple[np.array, np.array]:
        """
        Transforms image and target with image_index from nifti format into numpy array.
        :param global_config: preprocessing config
        :param image_index: index of image to be transformed
        :return: image and target as numpy arrays
        """
        link_config = global_config.get('transform_nifti_to_npy', None)

        if self.is_activated(link_config):
            image_file_path, target_file_path = self._get_files_names(image_index, 'nii.gz')
            npy_image = self._transform_nifti_to_npy(image_file_path)
            npy_target = self._transform_nifti_to_npy(target_file_path)
        else:
            image_file_path, target_file_path = self._get_files_names(image_index, 'npy')
            npy_image = np.load(image_file_path)
            npy_target = np.load(target_file_path)

        return npy_image, npy_target

    @staticmethod
    def _transform_nifti_to_npy(nifti_file_path: str) -> np.array:
        """
        Transforms single nifti file into the numpy array.
        :param nifti_file_path: path to nifti file
        :return: image or target in numpy array format
        """
        if os.path.isfile(nifti_file_path):
            nifti = nib.load(nifti_file_path)
            npy = nifti.get_fdata()
            return npy
        else:
            raise Exception(f'File {nifti_file_path} does not exist!')

    @staticmethod
    def _get_files_names(image_index: str, file_format: str) -> Tuple[str, str]:
        """
        On the base of the image index generates paths to image and target arrays.
        :param image_index: index of the image to be transformed
        :param file_format: format of the file with original image or target
        :return: paths to image and target to be transformed
        """
        image_file_path = os.path.join(DATA_DIR, IMAGES_DIR, f'IMG_{image_index}.{file_format}')
        target_file_path = os.path.join(DATA_DIR, REFERENCE_SEGMENTATIONS_DIR, f'LUNGS_IMG_{image_index}.{file_format}')
        return image_file_path, target_file_path
