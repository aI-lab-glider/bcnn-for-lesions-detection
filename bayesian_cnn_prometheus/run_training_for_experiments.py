from pathlib import Path
from itertools import product
import json
from os import mkdir
from typing import Any, Iterable, List, Optional
from dataclasses import dataclass
import operator
from functools import reduce
import copy
import os
from bayesian_cnn_prometheus.constants import Paths

@dataclass
class Override:
    key: str
    value: Any
    alias: Optional[str] = None

    def set_in_config(self, config):
        config = copy.deepcopy(config)
        path = self.key.split('.')
        parent = reduce(operator.getitem, path[:-1], config)
        parent[path[-1]] = self.value
        return config
    
    def __str__(self) -> str:
        if isinstance(self.value, list):
            value = '_'.join(str(i) for i in self.value)
        else:
            value = self.value
        key = self.alias or ".".join([segment[:3] for segment in self.key.split('.')])
        return f'{key}_{value}'


def create_config(weights_dir: str, overrides: Iterable[Override]):
    with open('bayesian_cnn_prometheus/config.local.json') as cfg:
        config = json.load(cfg)
    for override in overrides:
        config = override.set_in_config(config)
    config['weights_dir'] = weights_dir
    return config

def create_experiment_dir(override: Iterable[Override]):
    experiment_name = "_".join(str(item) for item in override)
    if not os.path.isdir(experiment_name):
        os.mkdir(experiment_name)
    experiment_dir = Path("tf2_experiments")/experiment_name
    if not experiment_dir.exists():
        experiment_dir.mkdir(parents=True)
    return experiment_dir

def save_experiment_config(experiment_config, experiment_dir):
    path_to_config = f'{experiment_dir}/config.json'
    with open(path_to_config, 'w+') as config_file:
        json.dump(experiment_config, config_file)
    return path_to_config


def run_tests(combinations_to_test):
    for combination in combinations_to_test:
        overrides = [[Override(key=override['key'], value=value, alias=override['alias']) for value in override['values']] for override in combination]
        for override in product(*overrides):
            experiment_dir = create_experiment_dir(override)
            experiment_config = create_config(str(experiment_dir), override)
            config_path = save_experiment_config(experiment_config, experiment_dir)
            sbatch_script_path = create_sbatch_script(experiment_dir) 
            os.system(f'sbatch {sbatch_script_path} {Paths.PROJECT_DIR/"main.py"} {config_path}')
            print('Submitted ', experiment_dir)



def create_sbatch_script(experiment_dir: Path):
    script_name = 'run_python_script'
    with open(f"{script_name}_TEMPLATE.sh", "r", encoding="utf-8") as f:
        contents = f.read()
        contents = contents.replace('BATCH_NAME', f'tf2_{experiment_dir}')
        contents = contents.replace("OUTPUT_FILE", f"{experiment_dir}/output.out")
        contents = contents.replace("ERROR_FILE", f"{experiment_dir}/error.err")
        contents = contents.replace("VENV_NAME", f"{experiment_dir}/venv")
    new_script_path = f'{experiment_dir}/{script_name}.sh'
    with open(new_script_path, "w", encoding="utf-8") as f:
        f.write(contents)
    return new_script_path


if __name__ == '__main__':
    combinations_to_test = [
    [
        {
            'alias': 'stride',
            'key': 'preprocessing.create_chunks.stride',
            'values': [
                [32,32,16],
                [64,64,32]

            ]
        },
        {
            'alias': 'shuffle',
            'key': 'preprocessing.create_chunks.should_shuffle',
            'values': [True]
        },
        ]
    ]
    run_tests(combinations_to_test)

