from typing import Dict

import numpy as np

from .chain_link import ChainLink


class CreateChunks(ChainLink):
    def run(self, global_config: Dict[str, str], array: np.array):
        link_config = global_config.get('create_chunks', None)
        if self.is_activated(link_config):
            chunk_size = tuple(link_config['chunk_size'])
            return self._create_chunks(array, chunk_size)

    @staticmethod
    def _create_chunks(data_subset: np.array, chunk_size: tuple = (32, 32, 16)):
        """
        Original data (numpy array) transformation into the array of 3d chunks.
        :param data_subset: single subset of data (or labels): np.array
        :param chunk_size: size of 3d chunk (a, b, c) to train the model with them: tuple
        """
        origin_x, origin_y, origin_z = data_subset.shape
        chunk_x, chunk_y, chunk_z = chunk_size

        for x in range(0, origin_x, chunk_x)[:-1]:
            for y in range(0, origin_y, chunk_y)[:-1]:
                for z in range(0, origin_z, chunk_z)[:-1]:
                    chunk = data_subset[x:x + chunk_x, y:y + chunk_y, z:z + chunk_z]
                    yield chunk
