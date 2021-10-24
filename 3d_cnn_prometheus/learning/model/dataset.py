import json
import os
from typing import Tuple

import numpy as np
import tensorflow as tf

from .constants import *
from .experiment_setup import ex
from .preprocessing_pipeline.preprocessing_pipeline import PreprocessingPipeline

cf = open('/home/alikrz/Pulpit/MyStuff/cancer-detection/3d-cnn-prometheus/3d_cnn_prometheus/learning/config_pp.json')
config = json.load(cf)
pp = PreprocessingPipeline(config)


def get_data_generator(dataset_type: DatasetType, batch_size: int = 1):
    """
    Generator for training or validation dataset.
    :param dataset_type: type of the dataset: DatasetType
    :param batch_size: size of batch for data (default: 1, for validation)
    :return data generator
    """
    return pp.run(dataset_type)


@ex.capture
def get_train_dataset(batch_size: int, chunk_size: tuple) -> tf.data.Dataset:
    """
    Create training dataset.
    :param batch_size: size of batch for train data
    :param chunk_size: size of chunks use for training
    :return training dataset
    """

    # train_data_generator = get_data_generator(str(CHUNKS_TRAIN_PATH), str(CHUNKS_TRAIN_TARGETS_PATH), batch_size,
    #                                           chunk_size)

    train_data_generator = pp.run(DatasetType.TRAIN_DIR, batch_size)

    # train_ds_output_shape = tf.TensorShape([batch_size, *chunk_size, 1])

    # TODO batch_size in generator, next line is tmp solution
    train_ds_output_shape = tf.TensorShape([1, *chunk_size, 1])
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

    # valid_data_generator = get_data_generator(str(CHUNKS_VALID_PATH), str(CHUNKS_VALID_TARGETS_PATH), 1, chunk_size)
    valid_data_generator = pp.run(DatasetType.VALID_DIR, 1)

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
