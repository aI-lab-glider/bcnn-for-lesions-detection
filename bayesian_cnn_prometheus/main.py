import json

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


if __name__ == '__main__':
    main()
