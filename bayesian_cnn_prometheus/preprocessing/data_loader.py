from typing import Generator

import tensorflow as tf

from bayesian_cnn_prometheus.preprocessing import DataGenerator


class DataLoader:

    def __init__(self, preprocessing_config, batch_size, chunk_size):
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.data_generator = DataGenerator(preprocessing_config, batch_size)
        self._train_data = None
        self._test_data = None
        self._valid_data = None

    def load_data(self):
        self._train_data = self._create_train_data()
        self._test_data = self._create_test_data()
        self._valid_data = self._create_valid_data()

    def get_train_data(self):
        return self._train_data

    def get_test_data(self):
        return self._test_data

    def get_valid_data(self):
        return self._valid_data

    def _create_train_data(self) -> tf.data.Dataset:
        generator = self.data_generator.get_train()
        return self._get_data_from_generator(generator)

    def _create_test_data(self) -> tf.data.Dataset:
        generator = self.data_generator.get_test()
        return self._get_data_from_generator(generator)

    def _create_valid_data(self) -> tf.data.Dataset:
        generator = self.data_generator.get_valid()
        return self._get_data_from_generator(generator)

    def _get_data_from_generator(self, generator: Generator) -> tf.data.Dataset:
        dataset_output_shape = tf.TensorShape([self.batch_size, *self.chunk_size, 1])
        dataset = tf.data.Dataset.from_generator(generator=generator, output_types=(tf.int64, tf.int64),
                                                 output_shapes=(dataset_output_shape, dataset_output_shape))
        return dataset
