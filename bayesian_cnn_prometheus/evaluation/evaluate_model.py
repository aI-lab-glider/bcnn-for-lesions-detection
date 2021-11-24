import json

from bayesian_cnn_prometheus.evaluation.bayesian_model_evaluator import BayesianModelEvaluator
from bayesian_cnn_prometheus.tools.results_visualizer import ResultsVisualizer


def main():
    weights_path = 'PATH_TO_WEIGHTS_H5'
    image_path = 'PATH_TO_SAMPLE_CT_SCAN'

    config = get_config()
    patient_id = image_path.split('.')[0].split('_')[-1]

    model_evaluator = BayesianModelEvaluator(weights_path, tuple([*config['chunk_size'], 1]))
    predictions = model_evaluator.evaluate(image_path, config['mc_samples'], [32, 32, 16])
    model_evaluator.save_predictions(patient_id, predictions)

    results_visualizer = ResultsVisualizer()
    results_visualizer.visualize_patient_results(patient_id, predictions, slice_number=83, save_variance=True)


def get_config():
    with open('../config.json') as cf:
        config = json.load(cf)
    return config


if __name__ == '__main__':
    main()
