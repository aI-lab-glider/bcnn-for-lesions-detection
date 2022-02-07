import functools
from optparse import Option
import random
from dataclasses import dataclass
from itertools import product
from typing import Callable, Dict, Generator, Iterable, Optional, Sequence, Tuple, List

import numpy as np

from bayesian_cnn_prometheus.constants import DatasetType
from bayesian_cnn_prometheus.evaluation.utils import standardize_image
from bayesian_cnn_prometheus.preprocessing.data_splitter import DataSplitter
from bayesian_cnn_prometheus.preprocessing.image_loader import ImageLoader
from batchgenerators.augmentations.color_augmentations import (
    augment_contrast, augment_brightness_multiplicative, augment_gamma)

from batchgenerators.augmentations.noise_augmentations import (augment_rician_noise,
                                                               augment_gaussian_noise)

from batchgenerators.augmentations.resample_augmentations import augment_linear_downsampling_scipy


AugmentationFunction = Callable[[np.array], np.array]


@dataclass
class DataGeneratorConfig:
    stride: Tuple[int, int, int]
    chunk_size: Tuple[int, int, int]
    should_shuffle: bool
    should_augment: bool = True


class DataGenerator:
    def __init__(self, preprocessing_config: Dict, batch_size: int):
        """
        Creates PreprocessingPipeline class to preprocess input data according to the config.
        :param preprocessing_config: structure with configuration to use in preprocessing in learning
        """
        self.batch_size = batch_size
        self.should_normalise = preprocessing_config["normalize_images"]["is_activated"]
        self.image_loader = ImageLoader(
            preprocessing_config['transform_nifti_to_npy']['ext'])

        self.config = config = DataGeneratorConfig(
            **preprocessing_config['create_chunks'])

        self.data_splitter = DataSplitter(preprocessing_config['create_data_structure'],
                                          preprocessing_config['update_healthy_patients_indices'])
        self.dataset_structure = self.data_splitter.split_indices()

        self._augmentations = self._create_augmentations() if config.should_augment else []

    def _create_augmentations(self) -> Sequence[AugmentationFunction]:
        def wrap_augmentation(augmentation_function: AugmentationFunction) -> AugmentationFunction:
            return lambda data: np.squeeze(
                augmentation_function(data.astype('float64')[None, ...]).astype('int16'),
                axis=0)
        return list(map(wrap_augmentation, [
            augment_contrast,
            augment_brightness_multiplicative,
            augment_gamma,
            augment_rician_noise,
            augment_gaussian_noise,
            augment_linear_downsampling_scipy,
        ]))

    def get_train(self):
        return self._get_data_generator(DatasetType.TRAIN, self.batch_size)

    def get_test(self):
        return self._get_data_generator(DatasetType.TEST, self.batch_size)

    def get_valid(self):
        return self._get_data_generator(DatasetType.VALID, self.batch_size)

    def _get_data_generator(self, dataset_type: str, batch_size: int):
        return functools.partial(self._generate_data, dataset_type, batch_size)

    def _generate_data(self, dataset_type: str, batch_size: int):
        """
        Creates a generator that produces arrays with chunks ready for training.
        :param dataset_type: type of dataset (train, test, valid)
        :param batch_size: number of chunks in one batch
        :return: generator that produces array with chunks
        """
        for x_npy, y_npy in self._image_flow(dataset_type):
            x_npy_norm = DataGenerator.normalize(x_npy) if self.should_normalise else x_npy
            images_chunks, targets_chunks = [], []

            for x_chunk, y_chunk in zip(self._generate_chunks(x_npy_norm, self.config.chunk_size, self.config.stride),
                                        self._generate_chunks(y_npy, self.config.chunk_size, self.config.stride)):
                x_chunk = x_chunk.reshape((*x_chunk.shape, 1))
                y_chunk = y_chunk.reshape((*y_chunk.shape, 1))

                images_chunks.append(x_chunk)
                targets_chunks.append(y_chunk)

                if len(images_chunks) == batch_size and len(targets_chunks) == batch_size:
                    if self.config.should_shuffle:
                        images_chunks, targets_chunks = self._shuffle_chunks(images_chunks, targets_chunks)
                    yield np.array(images_chunks), np.array(targets_chunks)

    def _image_flow(self, dataset_type: str):
        for image_index in self.dataset_structure[dataset_type]:
            x_npy, y_npy = self.image_loader.load(image_index)
            yield x_npy, y_npy

        augmentations = self._augmentations if not self.config.should_shuffle else random.sample(
            self._augmentations, len(self._augmentations))
        if not augmentations:
            yield None
        for image_index in self.dataset_structure[dataset_type]:
            augmentation = random.choice(augmentations)
            x_npy, y_npy = self.image_loader.load(image_index)
            x_npy_augmented, y_npy_augmented = augmentation(x_npy), augmentation(y_npy)
            yield x_npy_augmented, y_npy_augmented

    def _generate_chunks(self, dataset: np.ndarray, chunk_size: Tuple[int, int, int],
                         stride: Tuple[int, int, int]) -> Generator[np.ndarray, None, None]:
        """
        Generates chunks from the original data (numpy array).
        :param dataset: single subset of data (or labels)
        :param chunk_size: size of 3d chunk (a, b, c) to train the model with them
        :param stride: three-elements tuple with steps value to make in each axis
        :return: generator which produces chunks with size (a, b, c)
        """
        chunk_x, chunk_y, chunk_z = chunk_size

        for x, y, z in self._get_coords(dataset, chunk_size, stride):
            chunk = dataset[x:x + chunk_x, y:y + chunk_y, z:z + chunk_z]
            if chunk.shape == tuple(chunk_size):
                yield chunk

    @staticmethod
    def _shuffle_chunks(images_chunks: List[np.ndarray], targets_chunks: List[np.ndarray]) -> (
            List[np.ndarray], List[np.ndarray]):

        chunks = list(zip(images_chunks, targets_chunks))
        random.shuffle(chunks)
        images_chunks, targets_chunks = zip(*chunks)
        return list(images_chunks), list(targets_chunks)

    def _get_coords(self, dataset, chunk_size, stride):
        x_coords, y_coords, z_coords = [self._get_axis_coords_list(origin_shape, chunk_shape, stride)
                                        for origin_shape, chunk_shape, stride in zip(dataset.shape, chunk_size, stride)]

        for coords in product(x_coords, y_coords, z_coords):
            yield coords

    @staticmethod
    def _get_axis_coords_list(origin_shape, chunk_shape, stride):
        coords = list(range(chunk_shape, origin_shape - chunk_shape, stride))
        return coords

    @staticmethod
    def normalize(image: np.ndarray) -> np.ndarray:
        """
        Transforms data to have mean 0 and std 1 (standardize).
        :param image: non-standardized image to transform
        :return standardized image
        """
        return standardize_image(image)
