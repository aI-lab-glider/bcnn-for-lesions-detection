from typing import Dict, List
from preprocessing_pipeline.preprocessing_links.chain_link import ChainLink
import os
from shutil import copyfile


class CreateFolderStructure(ChainLink):
    def run(self, global_config: Dict[str, str]):
        link_config = global_config.get('create_folder_structure', None)
        if self.is_activated(link_config):
            prefixes = ['train', 'test', 'valid']
            config_keys = [(f'{prefix}_start', f'{prefix}_end')
                           for prefix in prefixes]
            ranges = [range(link_config[from_key], link_config[to_key])
                      for from_key, to_key in config_keys]
            destination_path: str = link_config['destination_path']

            def copy(to_dirs: List[str], data_from: str, data_prefix: str):
                self._copy_data(destination_path, to_dirs,
                                data_from, data_prefix, ranges)

            copy(to_dirs=prefixes,
                 data_from=link_config['imgs_path'], data_prefix='IMG')
            copy(to_dirs=[f'{prefix}_targets' for prefix in prefixes],
                 data_from=link_config['targets_path'], data_prefix='MASK')

    def _copy_data(self, destination_root: str, destination_leafs: List[str], data_from: str, data_prefix: str, ranges: List[range]):
        for dir_name, rng in zip(destination_leafs, ranges):
            os.mkdir(dir_name)
            from_paths = [f'{data_from}/{data_prefix}_{i:04d}' for i in rng]
            to_paths = [
                f'{destination_root}/{data_prefix}_{i:04d}' for i in rng]
            for from_p, to_p in zip(from_paths, to_paths):
                copyfile(from_p, to_p)
