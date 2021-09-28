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

    def _copy_data(self, destination_root: str, destination_subdirs: List[str], data_from: str, data_prefix: str, ranges: List[range]):
        """
        Copies .npy files from `data_from` to destination subdirs.
        Creates the following stucture:
        `destnation_root`
           |` destination_subdir[0]/
           |` destination_subdir[1]/
           |`...
            ` destination_subdir[N]/
        Contet of each destination subdir is specified be ranges.

        :param destination_root: name of directory to which data should be copied 
        :param destination_subdirs: list of directories to create in `destination_root` and to move data from `data_from` directories.
        :param data_from: directory from which data should be copied. Assumption is made, that dir have the following structure:
            `data_from`
               |` data_prefix_0001.npy
               |` data_prefix_0002.npy
               |`...
                ` data_prefix_N.npy

        :param data_prefix: 'MASK' or 'IMG'.
        :param ranges: list of ranges of len(destination_subdirs), which defines which elements should be copied        
        """
        if not os.path.isdir(destination_root):
            os.mkdir(destination_root)
        destinations = [
            f'{destination_root}/{leaf}' for leaf in destination_subdirs]
        for dir_name, rng in zip(destinations, ranges):
            if not os.path.isdir(dir_name):
                os.mkdir(dir_name)
            from_paths = [
                f'{data_from}/{data_prefix}_{i:04d}.npy' for i in rng]
            to_paths = [
                f'{dir_name}/{data_prefix}_{i:04d}.npy' for i in rng]
            for from_p, to_p in zip(from_paths, to_paths):
                copyfile(from_p, to_p)
