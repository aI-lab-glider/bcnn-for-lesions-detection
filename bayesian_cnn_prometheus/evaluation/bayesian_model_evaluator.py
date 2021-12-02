import nibabel as nib
from pathlib import Path
from typing import Tuple, List

import numpy as np
from tensorflow.python.keras.engine.training import Model
from tqdm import tqdm

from bayesian_cnn_prometheus.constants import Paths
from bayesian_cnn_prometheus.evaluation.utils import load_nifti_file, save_as_nifti, standardize_image
from bayesian_cnn_prometheus.learning.model.bayesian_vnet import bayesian_vnet
from bayesian_cnn_prometheus.utils import load_nifti_file, save_as_nifti


class BayesianModelEvaluator:
    def __init__(self, weights_path: str, chunk_size: Tuple = (32, 32, 16)):
        """
        Creates BayesianModelEvaluator object to evaluate bayesian model.
        :param weights_path: path to bayesian model weights in h5 format
        :param chunk_size: shape of the input to the model
        """
        self.chunk_size = chunk_size
        self.model = self.load_saved_model(weights_path)

    def evaluate(self, image_path: str, samples_num: int, stride: List[int]) -> List[np.ndarray]:
        """
        Samples model samples_num times and returns list of predictions.
        :param image_path: path to the image to predict
        :param samples_num: number of samples to make
        :param stride: three-elements list with steps value to make in each axis
        :return: list of arrays with samples_num predictions on image
        """
        image = load_nifti_file(image_path)
        image = standardize_image(image)
        image_chunks, coords = self._create_chunks(image, stride)
        predictions = []
        for _ in tqdm(range(samples_num)):
            prediction = np.zeros(image.shape)
            count_prediction = np.zeros(image.shape)

            for chunk, coord in zip(image_chunks, coords):
                reshaped_chunk = chunk.reshape(1, *chunk.shape, 1)
                chunk_pred = self.model.predict(reshaped_chunk)
                reshaped_chunk_pred = chunk_pred.reshape(*chunk.shape)

                prediction[self._get_window(coord)] += reshaped_chunk_pred
                count_prediction[self._get_window(
                    coord)] += np.ones(chunk.shape)
            predictions.append(prediction / np.maximum(count_prediction, 1))
        return predictions

    def _create_chunks(self, array: np.ndarray, stride: List[int]) -> Tuple[List[np.ndarray], List[Tuple[int, int, int]]]:
        """
        Generates chunks from the original data (numpy array).
        :param array: 3d array (image or reference or mask)
        :param stride: three-elements list with steps value to make in each axis
        :return: list of chunks and list of corresponding coordinates
        """
        origin_x, origin_y, origin_z = array.shape
        stride_x, stride_y, stride_z = stride

        chunks = []
        coords = []

        for x in range(0, origin_x, stride_x)[:-1]:
            for y in range(0, origin_y, stride_y)[:-1]:
                for z in range(0, origin_z, stride_z)[:-1]:
                    chunk = array[self._get_window((x, y, z))]
                    if chunk.shape == self.chunk_size[:3]:
                        chunks.append(chunk)
                        coords.append((x, y, z))

        return chunks, coords

    def _get_window(self, coord: Tuple[int, int, int]) -> Tuple[slice, ...]:
        return tuple([slice(dim_start, dim_start + chunk_dim) for (dim_start, chunk_dim) in zip(coord, self.chunk_size[:3])])

    @classmethod
    def save_predictions(cls, patient_id, predictions, affine: np.ndarray, nifti_header) -> None:
        """
        Saves predictions list in nifti file.
        :param patient_id: four-digit patient id
        :param predictions: list of arrays with predictions
        """
        predictions = np.array(predictions)
        segmentation = cls.get_segmentation_from_mean(predictions)
        variance = cls.get_segmentation_variance(predictions)
        predictions_path = str(Paths.SUMMARY_FILE_PATTERN_PATH).format(
            patient_id, 'nii.gz')
        save_as_nifti(segmentation, Path(
            predictions_path), affine, nifti_header)
        save_as_nifti(variance, Path(str(Paths.VARIANCE_FILE_PATTERN_PATH).format(
            patient_id, 'nii.gz')), affine, nifti_header)

    @staticmethod
    def get_segmentation_from_mean(predictions, threshold=0.463):
        segmentation = np.mean(predictions, axis=0)
        # segmentation[segmentation > threshold] = 1.
        # segmentation[segmentation <= threshold] = 0.
        return segmentation

    @staticmethod
    def get_segmentation_variance(predictions):
        return np.var(predictions, axis=0)

    def load_saved_model(self, weights_path: str) -> Model:
        """
        Loads bayesian model.
        :param weights_path: path to bayesian model weights in h5 format
        :return: keras model with loaded weights
        """
        model = bayesian_vnet((*self.chunk_size, 1),
                              kernel_size=3,
                              activation='relu',
                              padding='SAME',
                              prior_std=1)
        model.load_weights(weights_path)
        return model
