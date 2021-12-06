import os
from pathlib import Path

import click

from bayesian_cnn_prometheus.constants import Paths
from bayesian_cnn_prometheus.main import train_model


@click.command(name='train', help='Starts training of the model')
@click.option('--config-path', '--conf', '-c',
              default=Paths.CONFIG_PATH,
              type=click.STRING,
              help='Path to the configuration file')
@click.option('--data-path', '--data', '-d', '-t',
              default=Paths.DATA_DIR,
              type=click.STRING,
              help='Path to the data directory')
def start_training(config_path: str, data_path: str):
    print('Jon started submiting a train job ...')
    train_model(Path(config_path), Path(data_path))
    os.system(f'sbatch run_python_script.sh {Paths.PROJECT_DIR / "main.py"} {config_path} {data_path}')
