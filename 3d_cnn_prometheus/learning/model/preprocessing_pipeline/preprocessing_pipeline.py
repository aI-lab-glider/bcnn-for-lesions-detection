import glob
import os

import nibabel as nib
import numpy as np

from .preprocessing_links import TransformNiftiToNpy, NormalizeImages, CreateChunks
from ..constants import DatasetType, MASKS_DIR


class PreprocessingPipeline:
    def __init__(self, config):
        self.config = config
        self.patients_indices_to_train = self._get_indices_for_training()
        self.dataset_structure = self._split_indices()

    def run(self, dataset_type: DatasetType, batch_size: int = 1):
        def generator():
            steps = [TransformNiftiToNpy(self.config), NormalizeImages(self.config), CreateChunks(self.config)]
            for image_index in self.dataset_structure[dataset_type]:
                x_npy, y_npy = steps[0].run(self.config, image_index)
                x_npy_norm = steps[1].run(self.config, x_npy)
                images_chunks, targets_chunks = [], []
                for x_chunk, y_chunk in zip(steps[2].run(self.config, x_npy_norm), steps[2].run(self.config, y_npy)):
                    x_chunk = x_chunk.reshape((*x_chunk.shape, 1))
                    y_chunk = y_chunk.reshape((*y_chunk.shape, 1))

                    images_chunks.append(x_chunk)
                    targets_chunks.append(y_chunk)

                    if len(images_chunks) == batch_size and len(targets_chunks) == batch_size:
                        yield np.array(images_chunks), np.array(targets_chunks)

        return generator

    def _get_indices_for_training(self):
        masks_paths = glob.glob(os.path.join(self.config['transform_nifti_to_npy']['path_from'], MASKS_DIR, f'MASK_*'))
        healthy_masks_paths = [mask_path for mask_path in masks_paths if self._is_patient_healthy(mask_path)]
        healthy_patients_indices = [self._get_patient_index(mask_path) for mask_path in healthy_masks_paths]
        return healthy_patients_indices

    @staticmethod
    def _is_patient_healthy(mask_path: str) -> bool:
        nifti = nib.load(mask_path)
        npy = nifti.get_fdata()
        return len(np.unique(npy)) == 1

    @staticmethod
    def _get_patient_index(mask_path: str) -> str:
        return mask_path.split('.')[0].split('_')[1]

    def _split_indices(self) -> dict:
        patients_number = len(self.patients_indices_to_train)
        parts_sum = self.config['create_folder_structure']['train_part'] + self.config['create_folder_structure'][
            'valid_part'] + self.config['create_folder_structure']['test_part']

        valid_part = int(self.config['create_folder_structure']['valid_part'] * patients_number / parts_sum)
        valid_indices_len = max(valid_part, 1)

        test_part = int(self.config['create_folder_structure']['test_part'] * patients_number / parts_sum)
        test_indices_len = max(test_part, 1)

        train_indices_len = patients_number - valid_indices_len - test_indices_len

        return {
            DatasetType.TRAIN_DIR: self.patients_indices_to_train[:train_indices_len],
            DatasetType.VALID_DIR: self.patients_indices_to_train[
                                   train_indices_len: train_indices_len + valid_indices_len],
            DatasetType.TEST_DIR: self.patients_indices_to_train[train_indices_len + valid_indices_len:]
        }
