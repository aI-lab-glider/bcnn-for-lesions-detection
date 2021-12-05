import glob
import json
from pathlib import Path
from typing import Dict, List

import nibabel as nib
import numpy as np

import bayesian_cnn_prometheus
from bayesian_cnn_prometheus.constants import DatasetType, Paths
from bayesian_cnn_prometheus.utils import get_patient_index


class DataSplitter:

    def __init__(self, data_structure_config: Dict, should_update_healthy_patience_indices=True):
        # TODO ProxPxD - description
        """
        Creates PreprocessingPipeline class to preprocess input data according to the config.
        :param config: structure with configuration to use in preprocessing in learning
        """
        self.config = data_structure_config  # .get('preprocessing', {})  # ProxPxD
        self.should_update_healthy_patience_indices = should_update_healthy_patience_indices

    def split_indices(self) -> dict:
        """
        Divides indices into train, valid and test part.
        :return: dict with dataset types and their indices
        """
        patients_indices_to_train = self._get_indices_for_training()  # ProxPxD
        patients_number = len(patients_indices_to_train)
        parts_sum = self.config['train_part'] + self.config['valid_part'] + self.config['test_part']

        valid_part = int(self.config['valid_part'] * patients_number / parts_sum)
        valid_indices_len = max(valid_part, 1)

        test_part = int(self.config['test_part'] * patients_number / parts_sum)
        test_indices_len = max(test_part, 1)

        train_indices_len = patients_number - valid_indices_len - test_indices_len

        return {
            DatasetType.TRAIN: patients_indices_to_train[:train_indices_len],
            DatasetType.VALID: patients_indices_to_train[
                               train_indices_len: train_indices_len + valid_indices_len],
            DatasetType.TEST: patients_indices_to_train[train_indices_len + valid_indices_len:]
        }

    def _get_indices_for_training(self) -> List[str]:
        """
        Selects healthy patients to use their scans in this training.
        Uses healthy_patient_indices.json file as a cache.
        :return: list of patients indices
        """

        # ProxPxD idea: a file of paths that are being used in preprocessing module
        healthy_patients_indices_json_path = Path(
            bayesian_cnn_prometheus.preprocessing.__file__).parent / 'healthy_patients_indices.json'
        with open(healthy_patients_indices_json_path, 'r') as hf:
            healthy_patients_indices_json = json.load(hf)

        if self.should_update_healthy_patience_indices:
            healthy_patients_indices = self._find_healthy_patients_indices()

            healthy_patients_indices_json['healthy_patients_indices'] = healthy_patients_indices
            with open(healthy_patients_indices_json_path, 'w') as hf:
                json.dump(healthy_patients_indices_json, hf)
        else:
            healthy_patients_indices = healthy_patients_indices_json['healthy_patients_indices']

        return healthy_patients_indices

    def _find_healthy_patients_indices(self) -> List[str]:
        """
        Finds indices of healthy patients scans in dataset.
        :return: list of indices
        """
        masks_paths = glob.glob(str(Paths.MASK_FILE_PATTERN_PATH).format('*', 'nii.gz'))
        healthy_masks_paths = [target_path for target_path in masks_paths if self._is_patient_healthy(target_path)]
        healthy_patients_indices = [get_patient_index(mask_path) for mask_path in healthy_masks_paths]
        return healthy_patients_indices

    @staticmethod
    def _is_patient_healthy(target_path: str) -> bool:
        """
        Verifies if patient is healthy.
        :param target_path: path to the mask with pathological changes
        :return: the patient is healthy or not
        """
        nifti = nib.load(target_path)
        npy = nifti.get_fdata()
        return len(np.unique(npy)) == 1
