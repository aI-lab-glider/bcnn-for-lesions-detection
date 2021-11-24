import json

from bayesian_cnn_prometheus.constants import BATCH_SIZE, CHUNK_SIZE
from bayesian_cnn_prometheus.learning.bayesian_detector import BayesianDetector
from bayesian_cnn_prometheus.preprocessing.data_loader import DataLoader


def main():  # TODO ProxPxD find a proper name
    config = get_config()
    preprocessing_config = config.get('preprocessing')
    batch_size = config.get(BATCH_SIZE)  # 1config
    chunk_size = preprocessing_config.get('create_chunks').get(CHUNK_SIZE)  # 1config

    data_loader = DataLoader(config.get('preprocessing'), batch_size, chunk_size)
    data_loader.load_data()

    training_dataset = data_loader.get_train_data()
    validation_dataset = data_loader.get_valid_data()

    detector = BayesianDetector(config, batch_size, BayesianDetector.get_input_shape(training_dataset))
    detector.fit(training_dataset, validation_dataset)


def get_config():
    with open('config.json') as cf:
        config = json.load(cf)
    return config


if __name__ == '__main__':
    main()
