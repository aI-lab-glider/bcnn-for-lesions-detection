import json
from typing import Tuple

import tensorflow as tf

from bayesian_cnn_prometheus.constants import *
from bayesian_cnn_prometheus.learning.model.experiment_setup import ex
from bayesian_cnn_prometheus.preprocessing.data_generator import DataGenerator

# TODO Refactor this tmp solution
with open('./config.json') as cf:
    config = json.load(cf)
    preprocessing_pipeline = DataGenerator(config)


def get_data_generator(dataset_type: DatasetType, batch_size: int = 1):
    """
    Generator for training or validation dataset.
    :param dataset_type: type of the dataset: DatasetType
    :param batch_size: size of batch for data (default: 1, for validation)
    :return data generator
    """
    return preprocessing_pipeline._get_data_generator(dataset_type, batch_size)


@ex.capture
def get_train_dataset(batch_size: int, chunk_size: tuple) -> tf.data.Dataset:
    """
    Create training dataset.
    :param batch_size: size of batch for train data
    :param chunk_size: size of chunks use for training
    :return training dataset
    """
    train_data_generator = preprocessing_pipeline._get_data_generator(DatasetType.TRAIN, batch_size)

    train_ds_output_shape = tf.TensorShape([batch_size, *chunk_size, 1])
    train_ds = tf.data.Dataset.from_generator(generator=train_data_generator, output_types=(tf.int64, tf.int64),
                                              output_shapes=(train_ds_output_shape, train_ds_output_shape))

    return train_ds


@ex.capture
def get_valid_dataset(chunk_size: tuple) -> tf.data.Dataset:
    """
    Create validation dataset.
    :param chunk_size: size of chunks use for training
    :return training dataset
    """
    valid_data_generator = preprocessing_pipeline._get_data_generator(DatasetType.VALID, 1)

    valid_ds_output_shape = tf.TensorShape([1, *chunk_size, 1])
    valid_ds = tf.data.Dataset.from_generator(generator=valid_data_generator, output_types=(tf.int64, tf.int64),
                                              output_shapes=(valid_ds_output_shape, valid_ds_output_shape))
    return valid_ds


@ex.capture
def get_input_shape(dataset: tf.data.Dataset) -> Tuple[int]:
    """
    Get input dataset from tensorflow dataset object.
    :param dataset: tensorflow dataset object
    :return: input shape
    """
    return tuple([int(dim.value) for dim in list(dataset.element_spec[0].shape)])
