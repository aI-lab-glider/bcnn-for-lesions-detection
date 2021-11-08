import json
from typing import Tuple

import tensorflow as tf

from constants import *
from preprocessing.data_loader import DataLoader


# TODO ProxPxD find a proper name
def main():
    config: json = get_config()
    preprocessing_config = config.get('preprocessing')
    batch_size = config.get(BATCH_SIZE)
    chunk_size = preprocessing_config.get('create_chunks').get(CHUNK_SIZE)
    data_loader = DataLoader(preprocessing_config, batch_size, chunk_size)

    train_data = data_loader.get_train()
    test_data = data_loader.get_test()
    valid_data = data_loader.get_valid()


def get_config():
    with open('./config.json') as cf:
        config = json.load(cf)
    return config


def get_input_shape(dataset: tf.data.Dataset) -> Tuple[int]:
    """
    Get input dataset from tensorflow dataset object.
    :param dataset: tensorflow dataset object
    :return: input shape
    """
    return tuple([int(dim.value) for dim in list(dataset.element_spec[0].shape)])


if __name__ == '__main__':
    main()
