import json

from bayesian_cnn_prometheus.constants import BATCH_SIZE, CHUNK_SIZE
from bayesian_cnn_prometheus.learning.bayesian_detector import BayesianDetector
# TODO ProxPxD find a proper name
from bayesian_cnn_prometheus.preprocessing.data_loader import DataLoader


def main():
    config = get_config()
    preprocessing_config = config.get('preprocessing')
    batch_size = config.get(BATCH_SIZE)  # 1config
    chunk_size = preprocessing_config.get('create_chunks').get(CHUNK_SIZE)  # 1config

    data_loader = DataLoader(config.get('preprocessing'), batch_size, chunk_size)
    data_loader.load_data()

    detector = BayesianDetector(config)
    detector.put_train_data(data_loader.get_train_data())
    detector.put_valid_data(data_loader.get_valid_data())
    detector.create_model()
    detector.fit()


def get_config():
    with open('config.json') as cf:
        config = json.load(cf)
    return config


if __name__ == '__main__':
    main()
