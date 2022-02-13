import sys
from pathlib import Path

from bayesian_cnn_prometheus.analysis.lesions_analyzer import LesionsAnalyzer
from bayesian_cnn_prometheus.evaluation.evaluate_model import PredictionOptions
from bayesian_cnn_prometheus.evaluation.utils import load_config


def analyse_lesions():
    model_path = sys.argv[1]
    config_path = sys.argv[2]
    input_path = sys.argv[3]
    output_path = sys.argv[4]

    # patients_to_analysis = [3, 4, 5, 6, 8, 10, 11, 14, 17, 21, 26, 27, 29, 30, 32, 34, 35, 36, 37, 39, 41]
    patients_to_analysis = [3]

    model_config = load_config(Path(config_path))
    prediction_options = PredictionOptions(
        chunk_size=model_config['evaluation']['chunk_size'],
        stride=model_config['evaluation']['stride'],
        mc_sample=model_config['mc_samples'])

    lesions_analyzer = LesionsAnalyzer(model_path, input_path, prediction_options, patients_to_analysis, output_path)
    lesions_analyzer.run_analysis(should_postprocess_variance=True)


if __name__ == '__main__':
    analyse_lesions()
