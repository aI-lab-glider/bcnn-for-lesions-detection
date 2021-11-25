import json
import numpy as np

from bayesian_cnn_prometheus.evaluation.utils import load_nifti_file # refactor idea: move this function to more 'general' directory
from skimage import metrics
from matplotlib import pyplot as plt
from scipy.signal import correlate


class CorrelationAnalyzer:
    def __init__(self, variance_mask_path: str, cancer_mask_path: str):
        """
        Creates CorrelationAnalyzer object to compare generated variance masks with cancer masks.
        :param variance_mask_path: path to a file with generated variance masks
        :param cancer_mask_path: path to a file with original cancer masks
        """
        self.variance_mask_path = variance_mask_path
        self.cancer_mask_path = cancer_mask_path
        self.variance_mask = CorrelationAnalyzer.normalize(load_nifti_file(variance_mask_path))
        self.cancer_mask = CorrelationAnalyzer.normalize(load_nifti_file(cancer_mask_path))
        self.metrics = {}

    def perform_analysis(self, save_to_json: bool = False, print_metrics: bool = False, plot_metrics: bool = False):
        """
        Assigns the metrics describing cancer and variance correlation.
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
            print(f"- Adapted Rand error: {self.metrics['are']}")
            print(f"- precision: {self.metrics['precision']}")
            print(f"- recall: {self.metrics['recall']}")
            print(f"- cancer|variance masks conditional entropy: {self.metrics['cancer_to_variance_conditional_entropy']}")
            print(f"- variance|cancer masks conditional entropy: {self.metrics['variance_to_cancer_conditional_entropy']}")
            print(f"- the Hausdorff distance: {self.metrics['hausdorff_distance']}")
            print(f"- Mean Squared Error: {self.metrics['mse']}")
            print(f"- Normalized Root Mean Squared Error: {self.metrics['normalized_root_mse']}")
            print(f"- peak signal to noise ratio: {self.metrics['peak_signal_noise_ratio']}")
            # print(f"- mean structural similarity index: {self.metrics['ss_mean']}")

    def _plot_metrics(self):
        """
        Plots all of the displayable metrics.
        """
        if self.metrics:
            fig, axis = plt.subplots(1, 4)
            axis[0].plot(self.metrics["contingency_table"])
            axis[0].set_title("Contingency table")
            axis[1].plot(self.metrics["ss_image"])
            axis[1].set_title("Structural similarity image")
            axis[2].plot(self.metrics["ss_gradient"])
            axis[2].set_title("Structural similarity gradient")
            axis[3].plot(self.metrics["cross_correlate"])
            axis[3].set_title("Cross-correlate array")

    def _assign_adapted_rand_error_metrics(self):
        """
        Computes Adapted Rand error, precision and recall.
        """
        self.metrics["are"], self.metrics["precision"], self.metrics["recall"] = \
            metrics.adapted_rand_error(self.cancer_mask, self.variance_mask)

    def _assign_variation_of_information_metrics(self):
        """
        Computes variation of information between cancer and variance masks.
        """
        self.metrics["cancer_to_variance_conditional_entropy"], self.metrics["variance_to_cancer_conditional_entropy"] = \
            metrics.variation_of_information(self.cancer_mask, self.variance_mask)

    def _assign_hausdorff_distance(self):
        """
        Computes the Hausdorff distance of nonzero cancer and variance masks voxels.
        """
        self.metrics["hausdorff_distance"] = metrics.hausdorff_distance(self.cancer_mask, self.variance_mask)

    def _assign_mean_squared_error(self):
        """
        Computes Mean Squared Error between cancer and variance masks voxels.
        """
        self.metrics["mse"] = metrics.mean_squared_error(self.cancer_mask, self.variance_mask)

    def _assign_normalized_root_mse(self):
        """
        Computes Normalized Root Mean Squared Error between cancer and variance masks voxels.
        """
        self.metrics["normalized_root_mse"] = metrics.normalized_root_mse(self.cancer_mask, self.variance_mask)

    def _assign_peak_signal_noise_ratio(self):
        """
        Computes the peak signal to noise ratio between cancer and variance masks.
        """
        self.metrics["peak_signal_noise_ratio"] = metrics.peak_signal_noise_ratio(self.cancer_mask, self.variance_mask)

    def _assign_structural_similarity_metrics(self):
        """
        Computes structural similarity metrics between cancer and variance masks.
        """
        self.metrics["ss_mean"], self.metrics["ss_gradient"], self.metrics["ss_image"] = \
            metrics.structural_similarity(self.cancer_mask, self.variance_mask, gradient=True)

    def _assign_contingency_table(self):
        """
        Computes the contingency table of cancer and variance masks.
        """
        self.metrics["contingency_table"] = metrics.contingency_table(self.cancer_mask, self.variance_mask).toarray()
        
    def _assign_cross_correlate(self):
        """
        Computes the cross-correlate of cancer and variance masks.
        """
        self.metrics["cross_correlate"] = correlate(self.cancer_mask, self.variance_mask)

    # refactor idea: in the future, we should think about normalization in other places as well
    # alternatively, normalization should be performed once, e.g. directly before saving/after loading
    @staticmethod
    def normalize(arr: np.array):
        """
        Normalizes given array
        :param arr: numpy array
        """
        normalized_arr = (arr - arr.min()) / (arr.max() - arr.min())
        threshold = normalized_arr.max() / 10 # possible candidate to become a hyperparamether
        normalized_arr[normalized_arr > threshold] = 1
        normalized_arr[normalized_arr <= threshold] = 0

        return normalized_arr.astype('int')
