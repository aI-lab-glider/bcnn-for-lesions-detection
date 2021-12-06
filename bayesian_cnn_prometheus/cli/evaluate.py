import os

import click

from bayesian_cnn_prometheus.constants import Paths


@click.command(name='eval', help='Evaluates an image by the model')
@click.option('--config-path', '--conf', '-c',
              default=Paths.CONFIG_PATH,
              type=click.STRING,
              help='Path to the configuration file')
@click.option('--scan-path', '--scan', '-s', '-t',
              default=None,
              type=click.STRING,
              help='Path to the scan')
def evaluate(config_path, scan_path):
    os.system(
        f'sbatch run_python_script.sh {Paths.PROJECT_DIR / "evaluation " / "evaluate_model.py"} {config_path} {scan_path}')
