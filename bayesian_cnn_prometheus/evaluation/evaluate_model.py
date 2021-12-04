import json

from pathlib import Path
from bayesian_cnn_prometheus.constants import Paths
from bayesian_cnn_prometheus.evaluation.bayesian_model_evaluator import BayesianModelEvaluator
from bayesian_cnn_prometheus.evaluation.utils import load_nifti_file
import nibabel as nib


def main():
    patient_id = 3
    weights_path = str(Paths.PROJECT_DIR/'weights' /
                       'bayesian-03-0.814-1302760.h5')
    image_path = str(Paths.PROJECT_DIR.parent/'data' /
                     'IMAGES'/f'IMG_{patient_id:0>4}.nii.gz')

    config = get_config()

    image = nib.load(image_path)
    model_evaluator = BayesianModelEvaluator(
        weights_path, tuple([*config['preprocessing']['create_chunks']['chunk_size'], 1]))
    predictions = model_evaluator.evaluate(
        image_path, config['mc_samples'], config['preprocessing']['create_chunks']['stride'])
    model_evaluator.save_predictions(
        patient_id, predictions, image.affine, image.header)


def get_config():
    with open(Paths.PROJECT_DIR/'config.json') as cf:
        config = json.load(cf)
    return config


if __name__ == '__main__':
    main()
