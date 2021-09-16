from typing import Dict
from preprocessing_pipeline.preprocessing_links.chain_link import ChainLink
import os
import numpy as np


class CreateBatches(ChainLink):
    def run(self, global_config: Dict[str, str]):
        link_config = global_config.get('create_batches', None)
        if self.is_activated(link_config):
            self._create_batches(link_config['data_path'])

    def _create_batches(self, dir_path: str):
        """
        Original data (CT scans) transformation into numpy arrays of batches (inplace).
        Data directory should be organized as follow:

        data_path
            |--train
            |  |--train.npy
            |--test
            |  |--test.npy
            |--valid
            |  |--valid.npy
            |--train_targets
            |  |--train_targets.npy
            |--test_targets
            |  |--test_targets.npy
            |--valid_targets
            |  |--valid_targets.npy

        :param data_path: path to the directory with data divided into train, test and valid subsets: str
        """
        for subset_dir_name in os.listdir(dir_path):
            print(subset_dir_name)
            self.transform_single_subset_into_batches(
                dir_path, subset_dir_name)

    def transform_single_subset_into_batches(self, data_path: str, subset_dir_name: str):
        """
        Original data (single CT scan) transformation into numpy array of batches (inplace).

        :param data_path: path to the directory with data divided into train, test and valid subsets and theirs labels: str
        :param subset_dir_name: name of the data (or label) subset: str
        """
        for file_name in os.listdir(os.path.join(data_path, subset_dir_name)):
            subset_data_path = os.path.join(
                data_path, subset_dir_name, f'{file_name}')
            origin_subset_data = np.load(subset_data_path)

            batches_data = self.transform_3d_array_into_batches(
                origin_subset_data)
            batches_data = batches_data.reshape(*batches_data.shape, 1)

            np.save(subset_data_path, batches_data)

            origin_subset_data_path = os.path.join(
                data_path, subset_dir_name, f'origin_{file_name}')
            np.save(origin_subset_data_path, origin_subset_data)

    def transform_3d_array_into_batches(self, data_subset: np.array, batch_size: tuple = (32, 32, 16)) -> np.array:
        """
        Original data (numpy array) transformation into the array of 3d batches.
        :param data_subset: single subset of data (or labels): np.array
        :param batch_size: size of 3d batch (size: a, b, c) to train the model with them: tuple
        :return: numpy array of batches (size: X [number of batches], a, b, c): np.array
        """
        origin_x, origin_y, origin_z = data_subset.shape
        batch_x, batch_y, batch_z = batch_size

        batches = []

        for x in range(origin_x // batch_x)[:-1]:
            for y in range(origin_y // batch_y)[:-1]:
                for z in range(origin_z // batch_z)[:-1]:
                    batch = data_subset[x * batch_x:(x + 1) * batch_x, y * batch_y:(y + 1) * batch_y,
                                        z * batch_z:(z + 1) * batch_z]

                    batches.append(batch)

        return np.asarray(batches)
