from bayesian_cnn_prometheus.constants import Paths
from bayesian_cnn_prometheus.evaluation.utils import load_config

from bayesian_cnn_prometheus.learning.bayesian_detector import BayesianDetector
from bayesian_cnn_prometheus.preprocessing.data_loader import DataLoader


def main():
    config = load_config()
    preprocessing_config = config.get('preprocessing')
    batch_size = config.get('batch_size')
    chunk_size = preprocessing_config.get('create_chunks').get('chunk_size')

    data_loader = DataLoader(config.get(
        'preprocessing'), batch_size, chunk_size)
    data_loader.load_data()

    training_dataset = data_loader.get_train_data()
    validation_dataset = data_loader.get_valid_data()

    detector = BayesianDetector(
        config, batch_size, BayesianDetector.get_input_shape(training_dataset))
    detector.fit(training_dataset, validation_dataset)


if __name__ == '__main__':
    main()
