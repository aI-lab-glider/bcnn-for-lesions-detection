import os
from typing import Dict

import numpy as np

from .chain_link import ChainLink


class NormalizeImages(ChainLink):
    def run(self, global_config: Dict[str, str]):
        link_config = global_config.get('normalize_images', None)
        if self.is_activated(link_config):
            imgs_path = link_config['imgs_path']
            self._normalize_images(imgs_path)

    def _normalize_images(self, imgs_path: str) -> None:
        """
        For each image in directory apply normalizing operations
        and save with the same file name.
        :param imgs_path: path to directory with images
        :return:
        """
        for img_name in os.listdir(imgs_path):
            img_path = os.path.join(imgs_path, img_name)
            img = np.load(img_path)
            img_norm = self.standardize(img)
            np.save(img_path, img_norm)

    @staticmethod
    def standardize(data: np.array) -> np.array:
        """
        Transforms data to have mean 0 and std 1.
        :param data: non-standardized image to transform
        :return standardized image
        """
        return (data - np.mean(data)) / np.std(data)
