from typing import Tuple, List

import numpy as np
from tensorflow.python.keras.engine.training import Model
from tqdm import tqdm

from bayesian_cnn_prometheus.constants import Paths
from bayesian_cnn_prometheus.evaluation.utils import load_nifti_file, save_as_nifti
from bayesian_cnn_prometheus.learning.model.bayesian_vnet import bayesian_vnet


class BayesianModelEvaluator:
    def __init__(self, weights_path: str, input_shape: Tuple = (32, 32, 16, 1)):
        """
        Creates BayesianModelEvaluator object to evaluate bayesian model.
        :param weights_path: path to bayesian model weights in h5 format
        :param input_shape: shape of the input to the model
        """
        self.input_shape = input_shape
        self.model = self.load_saved_model(weights_path)

    def evaluate(self, image_path: str, samples_num: int, window: List[int]) -> List[np.array]:
        """
        Samples model samples_num times and returns list of predictions.
        :param image_path: path to the image to predict
        :param samples_num: number of samples to make
        :param window: three-elements list with steps value to make in each axis
        :return: list of arrays with samples_num predictions on image
        """
        image = load_nifti_file(image_path)
        image_chunks, coords = self.create_chunks(image, window)

        predictions = []
        for _ in tqdm(range(samples_num)):
            prediction = np.zeros(image.shape)
            count_prediction = np.zeros(image.shape)

            for chunk, coord in zip(image_chunks, coords):
                reshaped_chunk = chunk.reshape(1, *chunk.shape, 1)
                chunk_pred = self.model.predict(reshaped_chunk)
                reshaped_chunk_pred = chunk_pred.reshape(*chunk.shape)

                prediction[self._get_window(coord)] += reshaped_chunk_pred
                count_prediction[self._get_window(coord)] += np.ones(chunk.shape)

            predictions.append(prediction / np.maximum(count_prediction, 1))

        return predictions

    def _get_window(self, coord: List[int]) -> Tuple[slice, ...]:
        return tuple([slice(coord_, coord_ + input_shape_) for (coord_, input_shape_) in zip(coord, self.input_shape)])

    @staticmethod
    def save_predictions(patient_id, predictions) -> None:
        """
        Saves predictions list in nifti file.
        :param patient_id: four-digit patient id
        :param predictions: list of arrays with predictions
        """
        predictions_path = str(Paths.PREDICTIONS_FILE_PATTERN_PATH).format(patient_id, 'nii.gz')
        save_as_nifti(np.array(predictions), predictions_path)

    def load_saved_model(self, weights_path: str) -> Model:
        """
        Loads bayesian model.
        :param weights_path: path to bayesian model weights in h5 format
        :return: keras model with loaded weights
        """
        model = bayesian_vnet(self.input_shape,
                              kernel_size=3,
                              activation='relu',
                              padding='SAME',
                              prior_std=1)
        model.load_weights(weights_path)
        return model

    def create_chunks(self, array: np.array, window: List[int]) -> (List[np.array], List[Tuple[int, int, int]]):
        """
        Generates chunks from the original data (numpy array).
        :param array: 3d array (image or reference or mask)
        :param window: three-elements list with steps value to make in each axis
        :return: list of chunks and list of corresponding coordinates
        """
        origin_x, origin_y, origin_z = array.shape
        chunk_x, chunk_y, chunk_z, _ = self.input_shape
        window_x, window_y, window_z = window

        chunks = []
        coords = []

        for x in range(0, origin_x, window_x)[:-1]:
            for y in range(0, origin_y, window_y)[:-1]:
                for z in range(0, origin_z, window_z)[:-1]:
                    chunk = array[x:x + chunk_x, y:y + chunk_y, z:z + chunk_z]
                    if chunk.shape == self.input_shape[:3]:
                        chunks.append(chunk)
                        coords.append((x, y, z))

        return chunks, coords

    def add_chunk_to_array(self, array: np.array, chunk: np.array, coords: np.array) -> np.array:
        """
        Adds chunk to bigger array according to coordinates.
        :param array: big array
        :param chunk: small array, it should be added to the bigger one
        :param coords: coordinates of big array where small array should be added
        :return: big array with added small array
        """

        array[coords[0]:coords[0] + self.input_shape[0],
        coords[1]:coords[1] + self.input_shape[1],
        coords[2]:coords[2] + self.input_shape[2]] += chunk

        return array
