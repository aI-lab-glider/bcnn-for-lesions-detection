import argparse
import copy
import json
import operator
import os
from dataclasses import dataclass
from functools import reduce
from itertools import product
from pathlib import Path
from typing import Any, Iterable, Optional

from bayesian_cnn_prometheus.constants import Paths
from itertools import chain


EXPERIMENTS_DIR = Path('experiments')/'control_group'
EXPERIMENTS_DIR = str(EXPERIMENTS_DIR)



@dataclass
class ExperimentSetup:
    name: str
    overrides: Iterable['Override']

    @staticmethod
    def from_accumulated_dict(accumulated_config) -> Iterable['ExperimentSetup']:
        """
        Accumulated config is the config, that can have more than possible value for override
        {
            'name': ExpName,
            'overrides': [
            {
                'alias': 'Alias',
                'key': 'Key',
                'values': [val1, val2]
            },
            {
                'alias': 'Alias2',
                'key': 'Key',
                'values': [val1, val2]
            }],
            }
        }
        
        """
        overrides_for_keys = [
            [Override(key=override['key'], value=value, alias=override['alias']) for value in override['values']] 
            for override in accumulated_config['overrides']
            ]
        
        return [ExperimentSetup(name=f'{accumulated_config["name"]}', overrides=overrides).with_verbose_name() for overrides in product(*overrides_for_keys)]

    def with_verbose_name(self) -> 'ExperimentSetup':
        overrides_key = "_".join([str(o) for o in self.overrides])
        name = f'{self.name}_{overrides_key}'
        return ExperimentSetup(name=name, overrides=self.overrides)


    

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




def run_tests(experiments: Iterable[ExperimentSetup], is_local_execution):
    for experiment in experiments:
        experiment_dir = create_experiment_dir(experiment)
        experiment_config = create_config(str(experiment_dir), experiment.overrides)
        config_path = save_experiment_config(experiment_config, experiment_dir)
        sbatch_script_path = create_sbatch_script(experiment_dir)
        command = 'python' if is_local_execution else f'sbatch {sbatch_script_path}'
        os.system(f'{command} {Paths.PROJECT_DIR / "main.py"} {config_path}')
        print('Submitted ', experiment_dir)



def create_config(weights_dir: str, overrides: Iterable[Override]):
    with open('bayesian_cnn_prometheus/config.local.json') as cfg:
        config = json.load(cfg)
    for override in overrides:
        config = override.set_in_config(config)
    config['weights_dir'] = weights_dir
    return config


def create_experiment_dir(experiment: ExperimentSetup):
    experiment_name = experiment.name
    experiment_dir = Path(EXPERIMENTS_DIR) / experiment_name
    if not experiment_dir.exists():
        experiment_dir.mkdir(parents=True)
    return experiment_dir


def save_experiment_config(experiment_config, experiment_dir):
    path_to_config = f'{experiment_dir}/config.json'
    with open(path_to_config, 'w+') as config_file:
        json.dump(experiment_config, config_file)
    return path_to_config


def create_sbatch_script(experiment_dir: Path):
    script_name = 'run_python_script'
    with open(f"{script_name}_TEMPLATE.sh", "r", encoding="utf-8") as f:
        contents = f.read()
        contents = contents.replace('BATCH_NAME', f'{experiment_dir}')
        contents = contents.replace("OUTPUT_FILE", f"{experiment_dir}/output.out")
        contents = contents.replace("ERROR_FILE", f"{experiment_dir}/error.err")
        contents = contents.replace("VENV_NAME", f"{experiment_dir}/venv")
    new_script_path = f'{experiment_dir}/{script_name}.sh'
    with open(new_script_path, "w", encoding="utf-8") as f:
        f.write(contents)
    return new_script_path


@dataclass
class Args:
    is_local_execution: bool

    @staticmethod
    def parse():
        parser = argparse.ArgumentParser()
        parser.add_argument('-l', action='store_true', help='is training should be executed locally')
        args = parser.parse_args()
        return Args(args.l)


if __name__ == '__main__':
    stride_exp = {
        'name': 'stride_change', # Assumption: smaller stride will improve model quality because model will see more data,
        #  and more importantly it will see siimilar data in different contexts
        'overrides': [
            {
                'alias': 's',
                'key': 'preprocessing.create_chunks.stride',
                'values': [
                    [64, 8, 8],
                    [64, 16, 16], 
                    [128, 16, 16], 
                    [128, 32, 32]
                ]
            },
            {
                'alias': 'cs',
                'key': 'preprocessing.create_chunks.chunk_size',
                'values': [[128, 16, 16]]
            }
        ],
    }

    chunk_exp = {
        'name': 'chunk_change', # Assumption: bigger window will see more context and be able to get more precise results
        'overrides': [
            {
                'alias': 's',
                'key': 'preprocessing.create_chunks.chunk_size',
                'values': [
                    [4, 256, 4],
                    [8, 256, 8], 
                    [32, 64, 32], 
                    [8, 128, 32]
                ]
            },
            {
                'alias': 'cs',
                'key': 'preprocessing.create_chunks.stride',
                'values': [[16, 64, 16]]
            }
        ],
    }
    experiments = [chunk_exp]
    args = Args.parse()
    experiments = list(chain(*[ExperimentSetup.from_accumulated_dict(e) for e in experiments]))
    run_tests(experiments, args.is_local_execution)
