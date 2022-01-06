from pathlib import Path
from itertools import product
import json
from os import mkdir
from typing import Any, Iterable, List
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
        return f'{self.key}_{value}'


def create_config(experiment_name, overrides: Iterable[Override]):
    with open('bayesian_cnn_prometheus/config.local.json') as cfg:
        config = json.load(cfg)
    for override in overrides:
        config = override.set_in_config(config)
    config['weights_dir'] = experiment_name
    return config

def create_experiment_dir(override: Iterable[Override]):
    experiment_name = "_".join(str(item) for item in override)
    if not os.path.isdir(experiment_name):
        os.mkdir(experiment_name)
    return experiment_name

def save_config(test_config, experiment_name):
    path_to_config = f'{experiment_name}/config.json'
    with open(path_to_config, 'w+') as config_file:
        json.dump(test_config, config_file)
    return path_to_config


def run_tests(combinations_to_test):
    for combination in combinations_to_test:
        overrides = [[Override(key=override['key'], value=value) for value in override['values']] for override in combination]
        for override in product(*overrides):
            experiment_name = create_experiment_dir(override)
            test_config = create_config(experiment_name, override)
            config_path = save_config(test_config, experiment_name)
            print('Submitted ', experiment_name)
            os.system(f'sbatch run_python_script.sh {Paths.PROJECT_DIR/"main.py"} {config_path}')



if __name__ == '__main__':
    combinations_to_test = [
    [
        {
            'key': 'preprocessing.create_chunks.stride',
            'values': [
                [32,32,16],
                [8,8,4],
                [64,64,32]

            ]
        },
        {
            'key': 'preprocessing.create_chunks.should_shuffle',
            'values': [True, False]
        },
        ]
    ]
    run_tests(combinations_to_test)

