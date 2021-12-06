from pathlib import Path

import click

from bayesian_cnn_prometheus.constants import Paths
from bayesian_cnn_prometheus.evaluation.evaluate_model import evaluate_with_config_scan_id


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
    if scan_path is None:
        evaluate_with_config_scan_id(Path(config_path))
    else:
        evaluate(Path(config_path), scan_path)
