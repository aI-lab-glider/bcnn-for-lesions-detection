import random
from typing import Dict, List
from .chain_link import ChainLink
import os
from shutil import copyfile


class CreateFolderStructure(ChainLink):
    def run(self, global_config: Dict[str, str]):
        link_config = global_config.get('create_folder_structure', None)
        if self.is_activated(link_config):
            imgs_path = link_config['imgs_path']
            targets_path = link_config['targets_path']
            destination_path = link_config['destination_path']

            if not os.path.exists(destination_path):
                os.mkdir(destination_path)

            imgs_indices = [self._get_img_index(file_name) for file_name in os.listdir(link_config['imgs_path'])]

            if len(imgs_indices) < 3:
                raise Exception('You need more samples than subsets!')

            subsets = self._split_indices(imgs_indices, link_config)

            for (subset, indices) in subsets.items():
                self._copy_data(subset, indices, imgs_path, targets_path, destination_path)

    @staticmethod
    def _split_indices(imgs_indices: List[str], link_config: dict) -> dict:
        random.shuffle(imgs_indices)

        parts_sum = link_config['train_part'] + link_config['valid_part'] + link_config['test_part']

        valid_part = int(link_config['valid_part'] * len(imgs_indices) / parts_sum)
        valid_indices_len = valid_part if valid_part > 0 else 1

        test_part = int(link_config['test_part'] * len(imgs_indices) / parts_sum)
        test_indices_len = test_part if test_part > 0 else 1

        train_indices_len = len(imgs_indices) - valid_indices_len - test_indices_len

        return {
            'train': imgs_indices[:train_indices_len],
            'valid': imgs_indices[train_indices_len: train_indices_len + valid_indices_len],
            'test': imgs_indices[train_indices_len + valid_indices_len:]
        }

    @staticmethod
    def _get_img_index(img_file_name):
        return img_file_name.split('.')[0].split('_')[1]

    @staticmethod
    def _copy_data(subset, indices, imgs_path, targets_path, destination_path):
        if not os.path.isdir(destination_path):
            os.mkdir(destination_path)

        subset_path = os.path.join(destination_path, subset)
        if not os.path.isdir(subset_path):
            os.mkdir(subset_path)

        subset_targets_path = os.path.join(destination_path, f'{subset}_targets')
        if not os.path.isdir(subset_targets_path):
            os.mkdir(subset_targets_path)

        for idx in indices:
            img_file_name = f'IMG_{idx}.npy'
            img_path_src = os.path.join(imgs_path, img_file_name)
            img_path_dest = os.path.join(subset_path, img_file_name)
            copyfile(img_path_src, img_path_dest)

            target_file_name = f'MASK_{idx}.npy'
            target_path_src = os.path.join(targets_path, target_file_name)
            target_path_dest = os.path.join(subset_targets_path, target_file_name)
            copyfile(target_path_src, target_path_dest)
