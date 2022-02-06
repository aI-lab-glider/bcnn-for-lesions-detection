import os
from pathlib import Path

from bayesian_cnn_prometheus.constants import Paths

if __name__ == '__main__':
    experiments_catalogue = Paths.PROJECT_DIR.parent / Path('experiments')
    for experiment in experiments_catalogue.iterdir():
        experiment = Path(experiment)
        os.system(
            f'sbatch {experiments_catalogue / experiment / "run_python_script.sh"} '
            f'{Paths.PROJECT_DIR / "evaluation" / "evaluate_model.py"} {experiment}')
