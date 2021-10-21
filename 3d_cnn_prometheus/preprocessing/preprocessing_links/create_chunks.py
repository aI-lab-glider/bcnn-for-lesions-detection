import os
from typing import Dict

import numpy as np

from .chain_link import ChainLink


class CreateChunks(ChainLink):
    def run(self, global_config: Dict[str, str]):
        link_config = global_config.get('create_chunks', None)
        if self.is_activated(link_config):
            self._create_chunks(link_config['src_dir_path'], link_config['dst_dir_path'])

    def _create_chunks(self, src_dir_path: str, dst_dir_path: str):
        """
        Original data (CT scans) transformation into numpy arrays of chunks.
        Data directory should be organized as follow:

        data_path
            |--train
            |  |--IMG_0000.npy
            |  |--IMG_0001.npy
            |  |--...
            |--test
            |  |--IMG_0002.npy
            |  |--IMG_0003.npy
            |  |--...
            |--valid
            |  |--IMG_0004.npy
            |  |--IMG_0005.npy
            |  |--...
            |--train_targets
            |  |--MASK_0000.npy
            |  |--MASK_0001.npy
            |  |--...
            |--test_targets
            |  |--MASK_0002.npy
            |  |--MASK_0003.npy
            |  |--...
            |--valid_targets
            |  |--MASK_0004.npy
            |  |--MASK_0005.npy
            |  |--...

        :param src_dir_path: path to the directory with data divided into train, test and valid subsets: str
        :param dst_dir_path: path to the destination directory for chunks
        """
        if not os.path.exists(dst_dir_path):
            os.mkdir(dst_dir_path)

        for subset_dir_name in os.listdir(src_dir_path):
            os.mkdir(os.path.join(dst_dir_path, subset_dir_name))
            self._transform_single_subset_into_chunks(src_dir_path, dst_dir_path, subset_dir_name)

    def _transform_single_subset_into_chunks(self, src_dir_path: str, dst_dir_path: str, subset_dir_name: str):
        """
        Original data (single CT scan) transformation into numpy array of chunks.

        :param src_dir_path: path to the directory with data divided into train, test and valid subsets: str
        :param dst_dir_path: path to the destination directory for chunks
        :param subset_dir_name: name of the data (or label) subset: str
        """
        for file_name in os.listdir(os.path.join(src_dir_path, subset_dir_name)):
            subset_data_path = os.path.join(src_dir_path, subset_dir_name, file_name)
            origin_subset_data = np.load(subset_data_path)

            dst_file_prefix = os.path.join(dst_dir_path, subset_dir_name, file_name).split('.')[0]
            self._transform_3d_array_into_chunks(dst_file_prefix, origin_subset_data)

    def _transform_3d_array_into_chunks(self, dst_file_prefix: str, data_subset: np.array,
                                        chunk_size: tuple = (32, 32, 16)):
        """
        Original data (numpy array) transformation into the array of 3d chunks.
        :param data_subset: single subset of data (or labels): np.array
        :param chunk_size: size of 3d chunk (size: a, b, c) to train the model with them: tuple
        """
        origin_x, origin_y, origin_z = data_subset.shape
        chunk_x, chunk_y, chunk_z = chunk_size

        for x in range(origin_x // chunk_x)[:-1]:
            for y in range(origin_y // chunk_y)[:-1]:
                for z in range(origin_z // chunk_z)[:-1]:
                    chunk = data_subset[x * chunk_x:(x + 1) * chunk_x, y * chunk_y:(y + 1) * chunk_y,
                            z * chunk_z:(z + 1) * chunk_z]

                    chunk_path = f'{dst_file_prefix}_{x * chunk_x}_{y * chunk_y}_{z * chunk_z}.npy'
                    np.save(chunk_path, chunk)
