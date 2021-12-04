import numpy as np


def get_standardized_slice(array: np.array, slice_number: int) -> np.array:
    array = array[:, :, slice_number]
    array = array - np.min(array)
    divider = np.max(array) if np.max(array) > 0 else 1
    array = array * 255 / divider
    return array
