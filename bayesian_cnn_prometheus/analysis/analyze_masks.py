import sys

from dataclasses import dataclass, fields

from bayesian_cnn_prometheus.analysis.masks_analyzer import MasksAnalyzer
from bayesian_cnn_prometheus.constants import Paths
from bayesian_cnn_prometheus.evaluation.utils import assert_fields_have_values, load_config, get_arg


@dataclass
class MaskAnalysisConfig:
    model_name: str
    lesion_masks_path: str
    variance_masks_path: str


def analise(model_name: str, variance_masks_path: str, lesion_masks_path: str):
    config = MaskAnalysisConfig(model_name, lesion_masks_path, variance_masks_path)
    analise_from_config(config)


def analise_from_config(config: MaskAnalysisConfig):
    mask_analyzer = MasksAnalyzer(config.model_name, config.lesion_masks_path, config.variance_masks_path)
    mask_analyzer.perform_analysis(save_to_json=True)


def get_mask_analyser_config():
    app_config = load_config(Paths.CONFIG_PATH)
    assert_fields_have_values(app_config.get('mask_analysis', {}), [
        field.name for field in fields(MaskAnalysisConfig)])
    return MaskAnalysisConfig(**app_config['mask_analysis'])


if __name__ == '__main__':
    config = get_mask_analyser_config() if len(sys.argv) < 4 else MaskAnalysisConfig('', '', '')

    model_name = get_arg(1, config.model_name)
    lesion_masks_path = get_arg(2, config.lesion_masks_path)
    variance_masks_path = get_arg(3, config.variance_masks_path)

    analise(model_name, lesion_masks_path, variance_masks_path)
