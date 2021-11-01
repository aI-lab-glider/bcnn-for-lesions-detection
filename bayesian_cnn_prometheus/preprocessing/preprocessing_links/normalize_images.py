from typing import Dict

import numpy as np

from .chain_link import ChainLink


class NormalizeImages(ChainLink):
    def run(self, global_config: Dict[str, str], image: np.array) -> np.array:
        """
        Normalizes image.
        :param global_config: preprocessing config
        :param image: image to be normalized
        :return: normalized image
        """
        link_config = global_config.get('normalize_images', None)
        if self.is_activated(link_config):
            return self._normalize_images(image)
        else:
            return image

    @staticmethod
    def _normalize_images(image: np.array) -> np.array:
        """
        Transforms data to have mean 0 and std 1 (standardize).
        :param image: non-standardized image to transform
        :return standardized image
        """
        return (image - np.mean(image)) / np.std(image)
