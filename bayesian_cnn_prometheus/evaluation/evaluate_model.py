from pathlib import Path

import nibabel as nib
from dataclasses import dataclass, fields

from bayesian_cnn_prometheus.constants import Paths
from bayesian_cnn_prometheus.evaluation.bayesian_model_evaluator import BayesianModelEvaluator
from bayesian_cnn_prometheus.evaluation.utils import assert_fields_have_values, load_config


@dataclass
class EvaluationConfig:
    model_name: str
    patient_id: int


def evaluate_with_config_scan_id(config_path: Path):
    app_config = load_config(config_path)
    evaluation_config = EvaluationConfig(**app_config['evaluation'])
    scan_path = str(Paths.PROJECT_DIR.parent / 'data' /
                    'IMAGES' / f'IMG_{evaluation_config.patient_id:0>4}.nii.gz')
    evaluate(Paths.CONFIG_PATH, scan_path)


def evaluate(config_path: Path, scan_path: str):
    app_config = load_config(config_path)

    assert_fields_have_values(app_config.get('evaluation', {}), [
        field.name for field in fields(EvaluationConfig)])

    evaluation_config = EvaluationConfig(**app_config['evaluation'])

    weights_path = str(Paths.PROJECT_DIR.parent / 'weights' /
                       evaluation_config.model_name)
    # image_path = str(Paths.PROJECT_DIR.parent/'data' /
    #                  'IMAGES'/f'IMG_{evaluation_config.patient_id:0>4}.nii.gz')

    image = nib.load(scan_path)
    model_evaluator = BayesianModelEvaluator(
        weights_path,
        tuple([*app_config.get('preprocessing', {}
                               ).get('create_chunks', {}).get('chunk_size')])
    )

    predictions = model_evaluator.evaluate(
        scan_path,
        app_config.get('mc_samples'),
        app_config.get('preprocessing', {}).get('create_chunks', {}).get('stride'))

    model_evaluator.save_predictions(
        evaluation_config.patient_id,
        predictions,
        image.affine,
        image.header,
        app_config.get('should_perform_binarization', False))


if __name__ == '__main__':
    evaluate_with_config_scan_id(Paths.CONFIG_PATH)
