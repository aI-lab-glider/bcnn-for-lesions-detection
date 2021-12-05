import glob
import json
import os
from functools import reduce
from typing import List
from bayesian_cnn_prometheus.evaluation.utils import get_patient_index

from tqdm import tqdm

from bayesian_cnn_prometheus.analysis.similarity_comparer import SimilarityComparer
from bayesian_cnn_prometheus.constants import Paths, Metrics


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
        self.lesion_mask_paths = glob.glob(
            os.path.join(self.lesion_masks_path, Paths.MASK_FILE_PATTERN.format("*", "*")))
        self.variance_mask_paths = glob.glob(os.path.join(
            self.variance_masks_path, Paths.VARIANCE_FILE_PATTERN.format("*", "nii.gz")))
        self.results = {}

    def perform_analysis(self, save_to_json: bool = False):
        """
        Assigns similarity metrics describing each lesion and variance masks pair.
        :param save_to_json: if True, saves mean of corresponding metrics to JSON file
        """
        for path_to_lesion_mask, path_to_variance_mask in tqdm(zip(self.lesion_mask_paths, self.variance_mask_paths)):
            lesion_mask_name = os.path.basename(path_to_lesion_mask)
            pair_id = get_patient_index(lesion_mask_name)
            similarity_comparer = SimilarityComparer(
                path_to_lesion_mask, path_to_variance_mask)
            similarity_comparer.perform_analysis()
            metrics = similarity_comparer.metrics

            if metrics:
                self.results[pair_id] = metrics

        self._assign_metrics_means()

        if save_to_json:
            self._save_to_json()

    def _assign_metrics_means(self):
        """
        Calculates the means of every metrics respectively.
        """
        if self.results:
            means = {
                metric: 0.0 for metric in self.results[next(iter(self.results))]}
            cumulated_metrics = reduce(
                lambda metrics1, metrics2: {
                    metric: metrics1[metric] + metrics2[metric] for metric in means
                },
                self.results.values(),
                means)
            results_number = len(self.results)
            mean_metrics = {
                metric: cumulated_metrics[metric] / results_number for metric in cumulated_metrics}
            self.results[Metrics.MEANS] = mean_metrics

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
        Saves mean of corresponding metrics to a JSON file.
        """
        if self.results:
            with open(f"{self.model_name}_metrics.json", "w") as f:
                json.dump(self.results, f)
