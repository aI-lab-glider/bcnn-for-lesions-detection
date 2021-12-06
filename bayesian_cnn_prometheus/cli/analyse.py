import os

import click

from bayesian_cnn_prometheus.constants import Paths


@click.command(name='analyse')
@click.option('--lesion-masks-path', '-m',
              default=Paths.MASKS_PATH,
              type=click.STRING,
              help='Path to the lesion masks')
@click.option('--variance-masks-path', '-v',
              default=Paths.RESULTS_PATH,
              type=click.STRING,
              help='Path to the variance masks')
@click.option('--model-name')
def analyse(lesion_masks_path: str, variance_masks_path: str, model_name: str):
    os.system(
        f'sbatch run_python_script.sh {Paths.PROJECT_DIR / "analysis " / "analyze_masks.py"} '
        f'{model_name} {lesion_masks_path} {variance_masks_path}'
    )
