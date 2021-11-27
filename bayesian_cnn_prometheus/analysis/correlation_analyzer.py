import json
import numpy as np

from bayesian_cnn_prometheus.evaluation.utils import load_nifti_file # refactor idea: move this function to more 'general' directory
from skimage import metrics as sk_metrics
from matplotlib import pyplot as plt
from scipy.signal import correlate


class CorrelationAnalyzer:
    def __init__(self, variance_mask_path: str, lesion_mask_path: str):
        """
        Creates CorrelationAnalyzer object to compare generated variance mask with lesion mask.
        :param variance_mask_path: path to a file with generated variance mask
        :param lesion_mask_path: path to a file with original lesion mask
        """
        self.variance_mask_path = variance_mask_path
        self.lesion_mask_path = lesion_mask_path
        self.variance_mask = self.normalize(load_nifti_file(variance_mask_path))
        self.lesion_mask = self.normalize(load_nifti_file(lesion_mask_path))
        self.metrics = {}

    def perform_analysis(self, save_to_json: bool = False, print_metrics: bool = False, plot_metrics: bool = False):
        """
        Assigns the metrics describing lesion and variance correlation.
        :param save_to_json: if True, save metrics to JSON file
        :param print_metrics: if True, prints all of the numeric metrics
        :param plot_metrics: if True, plots all of the displayable metrics
        """
        self._assign_adapted_rand_error_metrics()
        self._assign_variation_of_information_metrics()
        self._assign_hausdorff_distance()
        self._assign_mean_squared_error()
        self._assign_normalized_root_mse()
        self._assign_peak_signal_noise_ratio()
        # self._assign_structural_similarity_metrics()
        # self._assign_contingency_table()
        # self._assign_cross_correlate()

        if save_to_json:
            self._save_to_json()

        if print_metrics:
            self._print_metrics()

        if plot_metrics:
            self._plot_metrics()

    def _save_to_json(self):
        """
        Saves metrics to a JSON file
        """
        if self.metrics:
            with open('metrics.json', 'w') as f:
                json.dump(self.metrics, f)

    def _print_metrics(self):
        """
        Prints all of the numeric metrics.
        """
        if self.metrics:
            print("Metrics:")
            numeric_metrics = [
                "are",
                "precision",
                "recall",
                "lesion_to_variance_conditional_entropy",
                "variance_to_lesion_conditional_entropy",
                "hausdorff_distance",
                "mse",
                "normalized_root_mse",
                "peak_signal_noise_ratio",
                # "ss_mean"
            ]

            for metric in numeric_metrics:
                print(f"- {metric} : {self.metrics[metric]}")

    def _plot_metrics(self):
        """
        Plots all of the displayable metrics.
        """
        if self.metrics:
            displayable_metrics = [
                "contingency_table",
                "ss_image",
                "ss_gradient",
                "cross_correlate",
            ]

            fig, axis = plt.subplots(1, len(displayable_metrics))

            for metric, ax in zip(displayable_metrics, axis):
                ax.plot(self.metrics[metric])
                ax.set_title(metric)

    def _assign_adapted_rand_error_metrics(self):
        """
        Computes Adapted Rand error, precision and recall.
        """
        self.metrics["are"], self.metrics["precision"], self.metrics["recall"] = \
            sk_metrics.adapted_rand_error(self.lesion_mask, self.variance_mask)

    def _assign_variation_of_information_metrics(self):
        """
        Computes variation of information between lesion and variance masks.
        """
        self.metrics["lesion_to_variance_conditional_entropy"], self.metrics["variance_to_lesion_conditional_entropy"] = \
            sk_metrics.variation_of_information(self.lesion_mask, self.variance_mask)

    def _assign_hausdorff_distance(self):
        """
        Computes the Hausdorff distance of nonzero lesion and variance masks voxels.
        """
        self.metrics["hausdorff_distance"] = sk_metrics.hausdorff_distance(self.lesion_mask, self.variance_mask)

    def _assign_mean_squared_error(self):
        """
        Computes Mean Squared Error between lesion and variance masks voxels.
        """
        self.metrics["mse"] = sk_metrics.mean_squared_error(self.lesion_mask, self.variance_mask)

    def _assign_normalized_root_mse(self):
        """
        Computes Normalized Root Mean Squared Error between lesion and variance masks voxels.
        """
        self.metrics["normalized_root_mse"] = sk_metrics.normalized_root_mse(self.lesion_mask, self.variance_mask)

    def _assign_peak_signal_noise_ratio(self):
        """
        Computes the peak signal to noise ratio between lesion and variance masks.
        """
        self.metrics["peak_signal_noise_ratio"] = sk_metrics.peak_signal_noise_ratio(self.lesion_mask, self.variance_mask)

    def _assign_structural_similarity_metrics(self):
        """
        Computes structural similarity metrics between lesion and variance masks.
        """
        self.metrics["ss_mean"], self.metrics["ss_gradient"], self.metrics["ss_image"] = \
            sk_metrics.structural_similarity(self.lesion_mask, self.variance_mask, full=True, gradient=True)

    def _assign_contingency_table(self):
        """
        Computes the contingency table of lesion and variance masks.
        """
        self.metrics["contingency_table"] = sk_metrics.contingency_table(self.lesion_mask, self.variance_mask).toarray()
        
    def _assign_cross_correlate(self):
        """
        Computes the cross-correlate of lesion and variance masks.
        """
        self.metrics["cross_correlate"] = correlate(self.lesion_mask, self.variance_mask)

    # refactor idea: in the future, we should think about normalization in other places as well
    # alternatively, normalization should be performed once, e.g. directly before saving/after loading
    @staticmethod
    def normalize(arr: np.array):
        """
        Normalizes given array
        :param arr: numpy array
        """
        normalized_arr = (arr - arr.min()) / (arr.max() - arr.min())
        threshold = normalized_arr.max() / 1000 # possible candidate to become a hyperparameter
        normalized_arr[normalized_arr > threshold] = 1
        normalized_arr[normalized_arr <= threshold] = 0

        return normalized_arr.astype('int')
