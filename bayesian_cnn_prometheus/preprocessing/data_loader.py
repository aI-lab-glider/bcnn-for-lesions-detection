from typing import Generator

import tensorflow as tf

from data_generator import DataGenerator


class DataLoader:

    def __init__(self, preprocessing_config, batch_size, chunk_size):
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.data_generator = DataGenerator(preprocessing_config, batch_size)

    def get_train(self):
        generator = self.data_generator.get_train()
        return self._get_data_from_generator(generator)

    def get_test(self):
        generator = self.data_generator.get_test()
        return self._get_data_from_generator(generator)

    def get_valid(self):
        generator = self.data_generator.get_valid()
        return self._get_data_from_generator(generator)

    def _get_data_from_generator(self, generator: Generator):
        dataset_output_shape = tf.TensorShape([self.batch_size, *self.chunk_size, 1])
        dataset = tf.data.Dataset.from_generator(generator=generator, output_types=(tf.int64, tf.int64),
                                                 output_shapes=(dataset_output_shape, dataset_output_shape))
        return dataset
