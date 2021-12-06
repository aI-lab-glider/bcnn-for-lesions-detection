import click

from bayesian_cnn_prometheus.analysis import analyze_masks
from bayesian_cnn_prometheus.constants import Paths


@click.command(name='analise')
@click.option('--lesion-masks-path', '-m',
              default=Paths.MASKS_PATH,
              type=click.STRING,
              help='Path to the lesion masks')
@click.option('--variance-masks-path', '-v',
              default=Paths.RESULTS_PATH,
              type=click.STRING,
              help='Path to the variance masks')
@click.option('--model-name')
def analise(lesion_masks_path: str, variance_masks_path: str, model_name: str):
    analyze_masks.analise(model_name, variance_masks_path, lesion_masks_path)
