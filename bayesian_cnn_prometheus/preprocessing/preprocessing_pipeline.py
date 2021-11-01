import glob
import os
from typing import Dict, Generator, List

import nibabel as nib
import numpy as np

from bayesian_cnn_prometheus.constants import DatasetType, DATA_DIR, MASKS_DIR
from bayesian_cnn_prometheus.preprocessing.preprocessing_links import TransformNiftiToNpy, NormalizeImages, CreateChunks


class PreprocessingPipeline:
    def __init__(self, global_config: Dict):
        """
        Creates PreprocessingPipeline class to preprocess input data according to the config.
        :param global_config: structure with configuration to use in preprocessing in learning
        """
        self.config = global_config.get('preprocessing', {})
        self.patients_indices_to_train = self._get_indices_for_training()
        self.dataset_structure = self._split_indices()

    def run(self, dataset_type: DatasetType, batch_size: int = 1) -> Generator:
        """
        Creates a generator that produces arrays with chunks ready for training.
        :param dataset_type: type of dataset (train, test, valid)
        :param batch_size: number of chunks in one batch
        :return: generator that produces array with chunks
        """

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

    def _get_indices_for_training(self) -> List[str]:
        """
        Select healthy patients to use their scans in this training.
        :return: list of patients indices
        """
        masks_paths = glob.glob(os.path.join(DATA_DIR, MASKS_DIR, f'MASK_*'))
        healthy_masks_paths = [target_path for target_path in masks_paths if self._is_patient_healthy(target_path)]
        healthy_patients_indices = [self._get_patient_index(mask_path) for mask_path in healthy_masks_paths]
        return healthy_patients_indices

    @staticmethod
    def _is_patient_healthy(target_path: str) -> bool:
        """
        Verify if patient is healthy.
        :param target_path: path to the mask with pathological changes
        :return: the patient is healthy or not
        """
        nifti = nib.load(target_path)
        npy = nifti.get_fdata()
        return len(np.unique(npy)) == 1

    @staticmethod
    def _get_patient_index(mask_path: str) -> str:
        """
        Extract patient index from the path to mask.
        :param mask_path: path to patient mask
        :return: patient index
        """
        return mask_path.split('.')[0].split('_')[-1]

    def _split_indices(self) -> dict:
        """
        Divide indices into train, valid and test part.
        :return: dict with dataset types and their indices
        """
        patients_number = len(self.patients_indices_to_train)
        parts_sum = self.config['create_data_structure']['train_part'] + self.config['create_data_structure'][
            'valid_part'] + self.config['create_data_structure']['test_part']

        valid_part = int(self.config['create_data_structure']['valid_part'] * patients_number / parts_sum)
        valid_indices_len = max(valid_part, 1)

        test_part = int(self.config['create_data_structure']['test_part'] * patients_number / parts_sum)
        test_indices_len = max(test_part, 1)

        train_indices_len = patients_number - valid_indices_len - test_indices_len

        return {
            DatasetType.TRAIN: self.patients_indices_to_train[:train_indices_len],
            DatasetType.VALID: self.patients_indices_to_train[
                               train_indices_len: train_indices_len + valid_indices_len],
            DatasetType.TEST: self.patients_indices_to_train[train_indices_len + valid_indices_len:]
        }
