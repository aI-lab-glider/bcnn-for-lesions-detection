import functools
import random
from itertools import product
from typing import Dict, Generator

import numpy as np

from bayesian_cnn_prometheus.constants import DatasetType
from bayesian_cnn_prometheus.evaluation.utils import standardize_image
from bayesian_cnn_prometheus.preprocessing.data_splitter import DataSplitter
from bayesian_cnn_prometheus.preprocessing.image_loader import ImageLoader


class DataGenerator:
    def __init__(self, preprocessing_config: Dict, batch_size: int):
        """
        Creates PreprocessingPipeline class to preprocess input data according to the config.
        :param preprocessing_config: structure with configuration to use in preprocessing in learning
        """
        self.batch_size = batch_size
        self.should_normalise = preprocessing_config.get(
            "normalize_images").get("is_activated")
        self.image_loader = ImageLoader(
            preprocessing_config.get('transform_nifti_to_npy').get('ext'))
        self.chunk_size = preprocessing_config.get(
            'create_chunks').get('chunk_size')
        self.stride = preprocessing_config.get('create_chunks').get('stride')
        self.data_splitter = DataSplitter(preprocessing_config.get('create_data_structure'),
                                          preprocessing_config.get('update_healthy_patients_indices'))
        self.dataset_structure = None

    def get_train(self):
        return self._get_data_generator(DatasetType.TRAIN, self.batch_size)

    def get_test(self):
        return self._get_data_generator(DatasetType.TEST, self.batch_size)

    def get_valid(self):
        return self._get_data_generator(DatasetType.VALID, self.batch_size)

    def _get_data_generator(self, dataset_type: DatasetType, batch_size: int):
        if self.dataset_structure is None:
            self.dataset_structure = self.data_splitter.split_indices()
        return functools.partial(self._generate_data, dataset_type, batch_size)

    def _generate_data(self, dataset_type: DatasetType, batch_size: int):
        """
        Creates a generator that produces arrays with chunks ready for training.
        :param dataset_type: type of dataset (train, test, valid)
        :param batch_size: number of chunks in one batch
        :return: generator that produces array with chunks
        """
        for image_index in self.dataset_structure[dataset_type]:
            x_npy, y_npy = self.image_loader.load(image_index)
            x_npy_norm = DataGenerator._normalize(
                x_npy) if self.should_normalise else x_npy
            images_chunks, targets_chunks = [], []
            for x_chunk, y_chunk in zip(self._generate_chunks(x_npy_norm, self.chunk_size, self.stride),
                                        self._generate_chunks(y_npy, self.chunk_size, self.stride)):
                x_chunk = x_chunk.reshape((*x_chunk.shape, 1))
                y_chunk = y_chunk.reshape((*y_chunk.shape, 1))

                images_chunks.append(x_chunk)
                targets_chunks.append(y_chunk)

                if len(images_chunks) == batch_size and len(targets_chunks) == batch_size:
                    yield np.array(images_chunks), np.array(targets_chunks)

    def _generate_chunks(self, data_subset: np.ndarray, chunk_size: tuple = (32, 32, 16),
                         stride: tuple = (16, 16, 8)) -> Generator[np.ndarray, None, None]:
        """
        Generates chunks from the original data (numpy array).
        :param data_subset: single subset of data (or labels)
        :param chunk_size: size of 3d chunk (a, b, c) to train the model with them
        :param stride: three-elements tuple with steps value to make in each axis
        :return: generator which produces chunks with size (a, b, c)
        """
        chunk_x, chunk_y, chunk_z = chunk_size

        for x, y, z in self._get_random_chunk_coords(data_subset, stride):
            chunk = data_subset[x:x + chunk_x, y:y + chunk_y, z:z + chunk_z]
            if chunk.shape == tuple(chunk_size):
                yield chunk

    def _get_random_chunk_coords(self, data_subset, stride):
        x_coords, y_coords, z_coords = [self._get_axis_coords_list(origin_shape, stride)
                                        for origin_shape, stride in zip(data_subset.shape, stride)]

        for coords in product(x_coords, y_coords, z_coords):
            yield coords

    @staticmethod
    def _get_axis_coords_list(origin_shape, stride):
        coords = list(range(stride, origin_shape - stride, stride))
        random.shuffle(coords)
        return coords

    @staticmethod
    def _normalize(image: np.ndarray) -> np.ndarray:
        """
        Transforms data to have mean 0 and std 1 (standardize).
        :param image: non-standardized image to transform
        :return standardized image
        """
        return standardize_image(image)
