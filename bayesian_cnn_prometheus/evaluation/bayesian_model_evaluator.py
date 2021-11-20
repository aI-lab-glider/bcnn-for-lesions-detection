from typing import Tuple, List
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.python.keras.engine.training import Model

from bayesian_cnn_prometheus.evaluation.utils import load_nifti_file
from bayesian_cnn_prometheus.learning.model.bayesian_vnet import bayesian_vnet


class BayesianModelEvaluator:
    def __init__(self, weights_path: str, input_shape: Tuple = (32, 32, 16, 1)):
        """
        Creates BayesianModelEvaluator object to evaluate bayesian model.
        :param weights_path: path to bayesian model weights in h5 format
        """
        self.input_shape = input_shape
        self.model = self.load_saved_model(weights_path)

    def evaluate(self, image_path: str, samples_num: int):
        image = load_nifti_file(image_path)
        image_chunks, coords = self.create_chunks(image)

        predictions = []
        for sample_num in tqdm(range(samples_num)):
            prediction = np.zeros(image.shape)

            for chunk, coord in zip(image_chunks, coords):
                reshaped_chunk = chunk.reshape(1, *chunk.shape, 1)
                chunk_pred = self.model.predict(reshaped_chunk)
                reshaped_chunk_pred = chunk_pred.reshape(*chunk.shape)
                prediction = self.add_chunk_to_array(prediction, reshaped_chunk_pred, coord)

            predictions.append(prediction)

        return predictions

    @staticmethod
    def get_segmentation_from_mean(predictions):
        segmentation = np.mean(predictions, axis=0)
        segmentation[segmentation > 0.5] = 1.
        segmentation[segmentation <= 0.5] = 0.
        return segmentation

    @staticmethod
    def get_segmentation_variance(predictions):
        return np.var(predictions, axis=0)

    @staticmethod
    def plot_segmentations(segmentation_from_mean: np.array, segmentation_variance: np.array, image_path, mask_path,
                           reference_path, slice_number):
        image = load_nifti_file(image_path)
        mask = load_nifti_file(mask_path)
        reference = load_nifti_file(reference_path)

        f, ax = plt.subplots(1, 5)
        ax[0] = image[:, :, slice_number]
        ax[1] = mask[:, :, slice_number]
        ax[2] = reference[:, :, slice_number]
        ax[3] = segmentation_from_mean[:, :, slice_number]
        ax[4] = segmentation_variance[:, :, slice_number]

        plt.show()

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

    def create_chunks(self, array: np.array) -> (List[np.array], List[Tuple[int, int, int]]):
        """
        Generates chunks from the original data (numpy array).
        :param array: 3d array (image or reference or mask)
        :return:
        """
        origin_x, origin_y, origin_z = array.shape
        chunk_x, chunk_y, chunk_z, _ = self.input_shape

        chunks = []
        coords = []

        for x in range(0, origin_x, chunk_x)[:-1]:
            for y in range(0, origin_y, chunk_y)[:-1]:
                for z in range(0, origin_z, chunk_z)[:-1]:
                    chunk = array[x:x + chunk_x, y:y + chunk_y, z:z + chunk_z]

                    chunks.append(chunk)
                    coords.append((x, y, z))

        return chunks, coords

    def reconstruct(self, chunks: List[np.array], coords: List[Tuple[int, int, int, int]],
                    origin_image_shape: Tuple[int, int]):
        """
        Reconstructs a 3D numpy array from generated chunks.
        :param chunks:
        :param coords:
        :param origin_image_shape:
        :return:
        """

        reconstructed_array = np.zeros(origin_image_shape)

        for chunk, coord in zip(chunks, coords):
            reconstructed_array[coord[0]:coord[0] + self.input_shape[0], coord[1]:coord[1] + self.input_shape[1],
            coord[2]:coord[2] + self.input_shape[2]] += chunk[0:2]

        return reconstructed_array

    def add_chunk_to_array(self, array, chunk, coords):
        """Adds a smaller 4D numpy array to a larger 4D numpy array."""

        array[coords[0]:coords[0] + self.input_shape[0],
        coords[1]:coords[1] + self.input_shape[1],
        coords[2]:coords[2] + self.input_shape[2]] += chunk

        return array


if __name__ == '__main__':
    weights_path = '/home/alikrz/Pulpit/MyStuff/cancer-detection/3d-cnn-prometheus/bayesian_cnn_prometheus/weights/bayesian-03-0.572-1319090.h5'
    bme = BayesianModelEvaluator(weights_path)

    image_path = '/home/alikrz/Pulpit/MyStuff/cancer-detection/3d-cnn-prometheus/bayesian_cnn_prometheus/data/IMAGES/IMG_0001.nii.gz'
    reference_path = '/home/alikrz/Pulpit/MyStuff/cancer-detection/3d-cnn-prometheus/bayesian_cnn_prometheus/data/REFERENCE_SEGMENTATIONS/LUNGS_IMG_0001.nii.gz'
    mask_path = '/home/alikrz/Pulpit/MyStuff/cancer-detection/3d-cnn-prometheus/bayesian_cnn_prometheus/data/MASKS/MASK_0001.nii.gz'

    predictions = bme.evaluate(image_path, 2)
    segmentations_from_mean = bme.get_segmentation_from_mean(predictions)
    segmentations_variance = bme.get_segmentation_variance(predictions)
    bme.plot_segmentations(segmentations_from_mean, segmentations_variance, image_path, mask_path, reference_path, 56)
