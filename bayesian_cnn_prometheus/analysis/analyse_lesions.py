from bayesian_cnn_prometheus.analysis.lesions_analyzer import LesionsAnalyzer
from bayesian_cnn_prometheus.evaluation.evaluate_model import PredictionOptions
from bayesian_cnn_prometheus.evaluation.utils import load_config


def analyse_lesions():
    model_path = '/Users/sol/Documents/3d-cnn-prometheus/experiments/normalization_sanity_check/old_norm/bayesian-15-0.878-9597.h5'
    input_path = '/Users/sol/Documents/3d-cnn-prometheus/bayesian_cnn_prometheus/analysis/data'
    output_path = '/Users/sol/Documents/3d-cnn-prometheus/bayesian_cnn_prometheus/analysis/data/RESULTS'

    model_config = load_config('/Users/sol/Documents/3d-cnn-prometheus/experiments/normalization_sanity_check/old_norm/config.json')
    prediction_options = PredictionOptions(
        chunk_size=model_config['preprocessing']['create_chunks']['chunk_size'],
        stride=model_config['preprocessing']['create_chunks']['stride'],
        mc_sample=model_config['mc_samples'])

    lesions_analyzer = LesionsAnalyzer(model_path, input_path, output_path, prediction_options)
    lesions_analyzer.run_analysis()


if __name__ == '__main__':
    analyse_lesions()
