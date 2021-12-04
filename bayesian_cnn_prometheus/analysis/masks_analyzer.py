import json
import os
import glob
import numpy as np

from bayesian_cnn_prometheus.analysis.similarity_comparer import SimilarityComparer
from bayesian_cnn_prometheus.constants import Paths, Metrics
from typing import List


class MasksAnalyzer:
    def __init__(self, model_name: str, lesion_masks_path: str, variance_masks_path: str):
        """
        Creates MasksAnalyzer object to compare generated variance masks with respective lesion masks.
        :param model_name: name of the model that has generated variance masks
        :param lesion_masks_path: path to a directory with original lesion masks
        :param variance_masks_path: path to a directory with generated variance masks
        """
        self.model_name = model_name
        self.lesion_masks_path = lesion_masks_path
        self.variance_masks_path = variance_masks_path
        self.lesion_mask_names = glob.glob(os.path.join(self.lesion_masks_path, Paths.MASK_FILE_PATTERN.format("*", "*")))
        self.variance_mask_names = glob.glob(os.path.join(self.variance_masks_path, "SEGMENTATION_VARIANCE_*.nii.gz"))
        self.metrics = {
            Metrics.DICE_COEFFICIENT: [],
            Metrics.HAUSDORFF_DISTANCE: [],
            Metrics.JACCARD_INDEX: [],
        }

    def perform_analysis(self, save_to_json: bool = False):
        """
        Assigns similarity metrics describing each lesion and variance masks pair.
        :param save_to_json: if True, saves mean of corresponding metrics to JSON file
        """
        for lesion_mask_name, variance_mask_name in zip(self.lesion_mask_names, self.variance_mask_names):
            path_to_lesion_mask = os.path.join(self.lesion_masks_path, lesion_mask_name)
            path_to_variance_mask = os.path.join(self.variance_masks_path, variance_mask_name)
            similarity_comparer = SimilarityComparer(path_to_lesion_mask, path_to_variance_mask)
            similarity_comparer.perform_analysis()
            metrics = similarity_comparer.metrics

            if metrics:
                for key in metrics:
                    self.metrics[key].append(metrics[key])

        if save_to_json:
            self._save_to_json()

    @staticmethod
    def load_mask_names(path_to_dir_with_mask_files: str) -> List[str]:
        """
        Loads mask paths from specified path.
        :param path_to_dir_with_mask_files: a path that mask paths will be loaded from
        :return: a list of sorted mask paths
        """
        return sorted(os.listdir(path_to_dir_with_mask_files))

    def _save_to_json(self):
        """
        Saves mean of corresponding metrics to a JSON file
        """
        if self.metrics:
            means = {key: np.mean(self.metrics[key]) for key in self.metrics}
            with open(self.model_name + "_metrics.json", "w") as f:
                json.dump(means, f)
