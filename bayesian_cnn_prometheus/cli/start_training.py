from pathlib import Path
import click
import os
from bayesian_cnn_prometheus.constants import Paths


@click.command()
def start_training():
    print('Jon started starting submit a traing job ...')
    os.system(f'sbatch run_python_script.sh {Paths.PROJECT_DIR/"main.py"}')
