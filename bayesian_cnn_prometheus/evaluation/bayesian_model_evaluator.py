from pathlib import Path
from typing import Tuple, List

import cv2 as cv
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from bayesian_cnn_prometheus.constants import Paths
from bayesian_cnn_prometheus.evaluation.utils import save_as_nifti, standardize_image
from bayesian_cnn_prometheus.learning.model.bayesian_vnet import BayesianVnet

Window = Tuple[int, int, int]
Stride = Tuple[int, int, int]


class BayesianModelEvaluator:
    def __init__(self, weights_path: str, chunk_size: Tuple = (32, 32, 16)):
        """
        Creates BayesianModelEvaluator object to evaluate bayesian model.
        :param weights_path: path to bayesian model weights in h5 format
        :param chunk_size: shape of the input to the model
        """
        self.chunk_size = chunk_size
        self.weights_path = weights_path

    def evaluate(self, image: np.ndarray, segmentation: np.ndarray, samples_num: int, stride: Stride,
                 should_binarize_prediction: bool = True) -> List[np.ndarray]:
        """
        Samples model samples_num times and returns list of predictions.
        :param image: image to predict
        :param segmentation: lungs segmentation mask
        :param samples_num: number of samples to make
        :param stride: three-elements list with steps value to make in each axis
        :param should_binarize_prediction: should prediction be binarized
        :return: list of arrays with samples_num predictions on image
        """
        image = standardize_image(image, segmentation)
        image_chunks, coords = self._create_chunks(image, stride)

        model = self.load_saved_model(self.weights_path, self.chunk_size)
        model = tf.function(model)

        predictions = []

        for _ in tqdm(range(samples_num)):
            prediction = np.zeros(image.shape)
            count_prediction = np.zeros(image.shape)

            for chunk, coord in zip(image_chunks, coords):
                reshaped_chunk = chunk.reshape(1, *chunk.shape, 1)
                chunk_pred = model(reshaped_chunk).numpy()
                reshaped_chunk_pred = chunk_pred.reshape(*chunk.shape)

                prediction[self._get_window(coord)] += reshaped_chunk_pred
                count_prediction[self._get_window(coord)] += np.ones(chunk.shape)

            prediction = prediction / np.maximum(count_prediction, 1)

            if should_binarize_prediction:
                prediction = self.binarize_prediction(prediction)

            predictions.append(prediction)

        return predictions

    def _create_chunks(self, array: np.ndarray, stride: Stride) -> Tuple[List[np.ndarray], List[Tuple[int, int, int]]]:
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
                    if chunk.shape == tuple(self.chunk_size[:3]):
                        chunks.append(chunk)
                        coords.append((x, y, z))

        return chunks, coords

    def _get_window(self, coord: Window) -> Tuple[slice, ...]:
        return tuple(
            [slice(dim_start, dim_start + chunk_dim) for (dim_start, chunk_dim) in zip(coord, self.chunk_size[:3])])

    @classmethod
    def save_predictions(
        cls, dir_path: Path, patient_idx: int, predictions, lungs_mask, affine: np.ndarray, nifti_header) -> None:
        """
        Saves predictions list in nifti file.
        :param patient_id: four-digit patient id
        :param predictions: list of arrays with predictions
        """
        predictions = np.array(predictions)
        segmentation = cls.get_segmentation_from_mean(predictions)
        variance = cls.get_segmentation_variance(predictions, lungs_mask)
        predictions_path = dir_path / Paths.PREDICTIONS_FILE_PATTERN.format(patient_idx, 'nii.gz')
        save_as_nifti(segmentation, predictions_path, affine, nifti_header)
        variance_path = dir_path / Paths.VARIANCE_FILE_PATTERN.format(patient_idx, 'nii.gz')
        save_as_nifti(variance, variance_path, affine, nifti_header)

    @staticmethod
    def get_segmentation_from_mean(predictions):
        segmentation = np.mean(predictions, axis=0)
        return segmentation

    @staticmethod
    def get_segmentation_variance(predictions, mask):
        predictions_variance = np.var(predictions, axis=0)
        predictions_variance[~mask.astype(bool)] = 0
        return predictions_variance

    def load_saved_model(self, weights_path: str, chunk_size: Window) -> BayesianVnet:
        """
        Loads bayesian model.
        :param weights_path: path to bayesian model weights in h5 format
        :param chunk_batch_shape: batch with all image chunks 
        :return: keras model with loaded weights
        """
        model = BayesianVnet((*chunk_size, 1),
                             kernel_size=3,
                             activation='relu',
                             padding='SAME',
                             prior_std=1)
        model.load_weights(weights_path)
        return model

    def binarize_prediction(self, mean_prediction: np.ndarray) -> np.ndarray:
        mean_prediction = mean_prediction.copy()
        mean_prediction = mean_prediction * 255
        mean_prediction = mean_prediction.astype(np.uint8)

        for i in range(mean_prediction.shape[0]):
            im_slice = mean_prediction[i]
            max_value = np.max(im_slice)
            mean_prediction[i, :, :] = cv.adaptiveThreshold(
                im_slice, max_value, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 129, 3)
        return mean_prediction
