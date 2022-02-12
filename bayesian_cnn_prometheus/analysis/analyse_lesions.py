import sys

from bayesian_cnn_prometheus.analysis.lesions_analyzer import LesionsAnalyzer
from bayesian_cnn_prometheus.evaluation.evaluate_model import PredictionOptions
from bayesian_cnn_prometheus.evaluation.utils import load_config


def analyse_lesions():
    model_path = sys.argv[1]
    config_path = sys.argv[2]
    input_path = sys.argv[3]

    model_config = load_config(config_path)
    prediction_options = PredictionOptions(
        chunk_size=model_config['preprocessing']['create_chunks']['chunk_size'],
        stride=model_config['preprocessing']['create_chunks']['stride'],
        mc_sample=model_config['mc_samples'])

    lesions_analyzer = LesionsAnalyzer(model_path, input_path, prediction_options)
    lesions_analyzer.run_analysis()


if __name__ == '__main__':
    analyse_lesions()
