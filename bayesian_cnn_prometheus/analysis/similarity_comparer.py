import numpy as np

from bayesian_cnn_prometheus.evaluation.utils import load_nifti_file # refactor idea: move this function to more 'general' directory
from skimage import metrics as sk_metrics
from typing import Dict


class SimilarityComparer:
    def __init__(self, lesion_mask_path: str, variance_mask_path: str):
        """
        Creates SimilarityComparer object to compare generated variance mask with lesion mask.
        :param lesion_mask_path: path to a file with original lesion mask
        :param variance_mask_path: path to a file with generated variance mask
        """
        self.lesion_mask_path = lesion_mask_path
        self.variance_mask_path = variance_mask_path
        self.lesion_mask = load_nifti_file(lesion_mask_path)
        self.variance_mask = self.normalize(load_nifti_file(variance_mask_path))
        self.metrics = {}

    def perform_analysis(self, print_metrics: bool = False):
        """
        Assigns the metrics describing lesion and variance similarity.
        :param print_metrics: if True, prints all of the numeric metrics
        """

        self._assign_dice_coefficient()
        self._assign_hausdorff_distance()
        self._assign_jaccard_index()

        if print_metrics:
            self._print_metrics()

    def get_metrics(self) -> Dict:
        """
        Returns similarity metrics.
        :return: a dictionary of metrics
        """
        if self.metrics:
            return self.metrics

    def _print_metrics(self):
        """
        Prints all of the numeric metrics.
        """
        if self.metrics:
            numeric_metrics = [
                "dice_coefficient",
                "hausdorff_distance",
                "jaccard_index",
            ]

            print("Metrics:")

            for metric in numeric_metrics:
                print(f"- {metric} : {self.metrics[metric]}")

    def _assign_dice_coefficient(self):
        """
        Computes the Dice similarity coefficient between lesion and variance masks.
        """
        changes_conjunction = np.sum(np.logical_and(self.lesion_mask, self.variance_mask))
        changes_count = np.sum(self.lesion_mask) + np.sum(self.variance_mask)
        self.metrics["dice_coefficient"] = \
            2 * changes_conjunction / changes_count if changes_count else .0

    def _assign_hausdorff_distance(self):
        """
        Computes the Hausdorff distance of nonzero lesion and variance masks voxels.
        """
        self.metrics["hausdorff_distance"] = sk_metrics.hausdorff_distance(self.lesion_mask, self.variance_mask)

    def _assign_jaccard_index(self):
        """
        Computes the Jaccard index between lesion and variance masks.
        """
        changes_conjunction = np.sum(np.logical_and(self.lesion_mask, self.variance_mask))
        changes_disjunction = np.sum(np.logical_or(self.lesion_mask, self.variance_mask))
        self.metrics["jaccard_index"] = \
            changes_conjunction / changes_disjunction if changes_disjunction else .0

    # refactor idea: in the future, we should think about normalization in other places as well
    # alternatively, normalization should be performed once, e.g. directly before saving/after loading
    @staticmethod
    def normalize(arr: np.array):
        """
        Normalizes given array
        :param arr: numpy array
        """
        normalized_arr = (arr - arr.min()) / (arr.max() - arr.min())
        threshold = normalized_arr.max() / 100  # possible candidate to become a hyperparameter
        normalized_arr[normalized_arr > threshold] = 1
        normalized_arr[normalized_arr <= threshold] = 0

        return normalized_arr.astype('int')
