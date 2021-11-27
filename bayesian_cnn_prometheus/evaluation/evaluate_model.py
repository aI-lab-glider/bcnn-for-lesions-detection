import json

from pathlib import Path
from bayesian_cnn_prometheus.evaluation.bayesian_model_evaluator import BayesianModelEvaluator
from bayesian_cnn_prometheus.tools.results_visualizer import ResultsVisualizer


def main():
    patient_id = 5
    weights_path = r'../weights/bayesian-03-0.962-1317496.h5'
    image_path = f'../../data/IMAGES/IMG_{patient_id:0>4}.nii.gz'

    config = get_config()

    model_evaluator = BayesianModelEvaluator(
        weights_path, tuple([*config['preprocessing']['create_chunks']['chunk_size'], 1]))
    predictions = model_evaluator.evaluate(
        image_path, config['mc_samples'], config['preprocessing']['create_chunks']['stride'])
    model_evaluator.save_predictions(patient_id, predictions)

    results_visualizer = ResultsVisualizer()
    results_visualizer.visualize_patient_results(
        patient_id, predictions, slice_number=83, save_variance=True)


def get_config():
    with open('../config.json') as cf:
        config = json.load(cf)
    return config


if __name__ == '__main__':
    main()
