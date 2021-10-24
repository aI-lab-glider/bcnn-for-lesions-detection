import os
from typing import Dict

import nibabel as nib
import numpy as np

from .chain_link import ChainLink
from ...constants import IMAGES_DIR, MASKS_DIR


class TransformNiftiToNpy(ChainLink):
    def run(self, global_config: Dict[str, str], image_index: str):
        link_config = global_config.get('transform_nifti_to_npy', None)

        if self.is_activated(link_config):
            image_file_path, target_file_path = self._get_files_names(link_config, image_index, 'nii.gz')
            npy_image = self._transform_nifti_to_npy(image_file_path)
            npy_target = self._transform_nifti_to_npy(target_file_path)
        else:
            image_file_path, target_file_path = self._get_files_names(link_config, image_index, 'npy')
            npy_image = np.load(image_file_path)
            npy_target = np.load(target_file_path)

        return npy_image, npy_target

    @staticmethod
    def _transform_nifti_to_npy(nifti_file_path: str):
        if os.path.isfile(nifti_file_path):
            nifti = nib.load(nifti_file_path)
            npy = nifti.get_fdata()
            return npy
        else:
            raise Exception(f'File {nifti_file_path} does not exist!')

    @staticmethod
    def _get_files_names(link_config: dict, image_index: str, file_format: str):
        image_file_path = os.path.join(link_config['path_from'], IMAGES_DIR, f'IMG_{image_index}.{file_format}')
        target_file_path = os.path.join(link_config['path_from'], MASKS_DIR, f'MASK_{image_index}.{file_format}')
        return image_file_path, target_file_path
