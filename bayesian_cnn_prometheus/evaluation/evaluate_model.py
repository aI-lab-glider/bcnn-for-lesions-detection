import json

from pathlib import Path
from bayesian_cnn_prometheus.constants import Paths
from bayesian_cnn_prometheus.evaluation.bayesian_model_evaluator import BayesianModelEvaluator
import nibabel as nib
from dataclasses import dataclass, fields

from bayesian_cnn_prometheus.evaluation.utils import assert_fields_have_values, load_config


@dataclass
class EvaluationConfig:
    model_name: str
    patient_id: int


def main():
    app_config = load_config()

    assert_fields_have_values(app_config.get('evaluation', {}), [
                              field.name for field in fields(EvaluationConfig)])

    evaluation_config = EvaluationConfig(**app_config['evaluation'])

    weights_path = str(Paths.PROJECT_DIR.parent/'weights' /
                       evaluation_config.model_name)
    image_path = str(Paths.PROJECT_DIR.parent/'data' /
                     'IMAGES'/f'IMG_{evaluation_config.patient_id:0>4}.nii.gz')

    image = nib.load(image_path)
    model_evaluator = BayesianModelEvaluator(
        weights_path,
        tuple([*app_config.get('preprocessing', {}
                               ).get('create_chunks', {}).get('chunk_size')])
    )

    predictions = model_evaluator.evaluate(
        image_path,
        app_config.get('mc_samples'),
        app_config.get('preprocessing', {}).get('create_chunks', {}).get('stride'))

    model_evaluator.save_predictions(
        evaluation_config.patient_id,
        predictions,
        image.affine,
        image.header,
        app_config.get('should_perform_binarization', False))


if __name__ == '__main__':
    main()
