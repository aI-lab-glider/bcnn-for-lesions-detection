import numpy as np


class ImageNormalizer:

    @staticmethod
    def normalize(image: np.array) -> np.array:
        """
        Transforms data to have mean 0 and std 1 (standardize).
        :param image: non-standardized image to transform
        :return standardized image
        """
        return (image - np.mean(image)) / np.std(image)
