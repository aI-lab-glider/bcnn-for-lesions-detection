import numpy as np
from matplotlib import pyplot as plt

from bayesian_cnn_prometheus.constants import Paths
from bayesian_cnn_prometheus.evaluation.utils import load_nifti_file, get_standardized_slice


class ResultsVisualizer:
    def __init__(self):
        ...

    def visualize_patient_results(self, patient_id: str, predictions: np.array, slice_number: int):
        image_file_path = str(Paths.IMAGE_FILE_PATTERN_PATH).format(patient_id, 'nii.gz')
        mask_file_path = str(Paths.MASK_FILE_PATTERN_PATH).format(patient_id, 'nii.gz')
        reference_file_path = str(Paths.REFERENCE_SEGMENTATION_FILE_PATTERN_PATH).format(patient_id, 'nii.gz')

        image = load_nifti_file(image_file_path)
        mask = load_nifti_file(mask_file_path)
        reference = load_nifti_file(reference_file_path)

        segmentation_from_mean = self.get_segmentation_from_mean(predictions)
        segmentation_variance = self.get_segmentation_variance(predictions)

        self.plot_segmentations(image, reference, mask, segmentation_from_mean, segmentation_variance, patient_id,
                                slice_number)

    @staticmethod
    def plot_segmentations(image, reference, mask, segmentation_from_mean: np.array, segmentation_variance: np.array,
                           patient_id, slice_number):
        fig = plt.figure(constrained_layout=False)
        fig.suptitle(f'Patient: {patient_id}, slice number: {slice_number}', fontsize=14, fontweight='bold')
        fig.set_figheight(10)
        fig.set_figwidth(10)

        grid = fig.add_gridspec(nrows=6, ncols=4, left=0.1, right=0.9, wspace=0.1, hspace=0.4)

        image_ax = fig.add_subplot(grid[:2, 1:3])
        image_ax.set_title('CT Scan')
        image_ax.imshow(get_standardized_slice(image, slice_number), cmap='gray')
        image_ax.axis('off')

        reference_ax = fig.add_subplot(grid[2:4, :2])
        reference_ax.set_title('Reference Segmentation')
        reference = get_standardized_slice(reference, slice_number)
        reference_ax.imshow(reference, cmap='gray')
        reference_ax.axis('off')

        segmentation_from_mean_ax = fig.add_subplot(grid[2:4, 2:])
        segmentation_from_mean_ax.set_title('Predicted Segmentation Mean')
        segmentation_from_mean_ax.imshow(get_standardized_slice(segmentation_from_mean, slice_number), cmap='gray')
        segmentation_from_mean_ax.axis('off')

        mask_ax = fig.add_subplot(grid[4:, :2])
        mask_ax.set_title('Reference Mask')
        mask_ax.imshow(get_standardized_slice(mask, slice_number), cmap='gray')
        mask_ax.axis('off')

        segmentation_variance_ax = fig.add_subplot(grid[4:, 2:])
        segmentation_variance_ax.set_title('Predicted Segmentation Variance')
        segmentation_variance = get_standardized_slice(segmentation_variance, slice_number)
        segmentation_variance_masked = np.multiply(segmentation_variance, np.array(reference, dtype=bool))
        segmentation_variance_ax.imshow(segmentation_variance_masked, cmap='gray')
        segmentation_variance_ax.axis('off')

        plt.savefig(str(Paths.SUMMARY_FILE_PATTERN_PATH).format(patient_id, str(slice_number), 'png'))

    @staticmethod
    def get_segmentation_from_mean(predictions):
        segmentation = np.mean(predictions, axis=0)
        segmentation[segmentation > 0.2] = 1.
        segmentation[segmentation <= 0.2] = 0.
        return segmentation

    @staticmethod
    def get_segmentation_variance(predictions):
        return np.var(predictions, axis=0)
