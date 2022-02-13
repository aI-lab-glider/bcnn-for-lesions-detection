import os
import sys
from pathlib import Path

from bayesian_cnn_prometheus.constants import Paths
from bayesian_cnn_prometheus.run_training_for_experiments import create_sbatch_script


def run_analysis_for_experiment():
    model_path = sys.argv[1]
    config_path = sys.argv[2]
    input_path = sys.argv[3]
    output_path = sys.argv[4]

    script_path = create_sbatch_script(Path(output_path))
    os.system(
        f'sbatch {script_path} {os.path.join(Paths.PROJECT_DIR, "analysis", "analyse_lesions.py")} {model_path} {config_path} {input_path} {output_path}')


if __name__ == '__main__':
    run_analysis_for_experiment()