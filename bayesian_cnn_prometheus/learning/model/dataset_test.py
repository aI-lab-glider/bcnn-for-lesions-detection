import os
import pickle

import numpy as np

from .experiment_setup import ex
from .utils import absolute_file_paths, standardize


@ex.capture
def create_chunks(arr: np.ndarray, batch_size: int, num_gpus: int, step: int, chunk_size: list, trim: bool = True):
    """Chunks a 4D numpy array into smaller 4D arrays."""

    chunks = []
    coords = []

    if len(chunk_size) != 3 or len(arr.shape) != 3:
        raise ValueError("Wrong dimensions!")
    
    z_size, x_size, y_size = chunk_size
    z_max, x_max, y_max = np.asarray(arr.shape) - chunk_size

    if z_max < 0 or x_max < 0 or y_max < 0:
        raise ValueError("Volume is too small for the given chunk size.")

    for z in range(0, z_max, z_size // step):
        for x in range(0, x_max, x_size // step):
            for y in range(0, y_max, y_size // step):
                coords.append((z, x, y))
                chunks.append(arr[z:z + z_size, x:x + x_size, y:y + y_size, :])


    chunks = np.asarray(chunks)

    # Avoids https://github.com/keras-team/keras/issues/11434
    if trim:
        last_batch_gpus = (chunks.shape[0] % batch_size) % num_gpus
        if last_batch_gpus != 0:
            chunks = chunks[:-last_batch_gpus, :, :, :, :]
            coords = coords[:-last_batch_gpus]

    return chunks, coords, arr.shape


@ex.capture
def reconstruct(arr, coords, shape, window):
    """Reconstructs a 4D numpy array from its generated chunks."""

    new = np.zeros(shape)
    count = np.zeros(shape)

    for chunk, coord in zip(arr, coords):
        new[coord[0]:coord[0] + window[0],
        coord[1]:coord[1] + window[1],
        coord[2]:coord[2] + window[2], :] += chunk

        count[coord[0]:coord[0] + window[0],
        coord[1]:coord[1] + window[1],
        coord[2]:coord[2] + window[2], :] += 1.

    return new / count


def add_chunk_to_arr(arr, chunk, coords, shape):
    """Adds a smaller 4D numpy array to a larger 4D numpy array."""

    arr[coords[0]:coords[0] + shape[0],
    coords[1]:coords[1] + shape[1],
    coords[2]:coords[2] + shape[2], :] += chunk

    return arr


@ex.capture
def load_data(files, vnet, batch_size, num_gpus, norm):
    """Loads and preprocesses data."""

    # Optionally standardizes data.
    if norm:
        arr = [standardize(np.load(file)) for file in files]
    else:
        arr = [np.load(file) for file in files]

    if len(arr) == 1:
        arr = arr[0]
    # If all the same shape, concat.
    elif len(set([sub_arr.shape for sub_arr in arr])) == 1:
        arr = np.concatenate(arr)
    # If different shapes and 3D, chunk then concat.
    elif vnet:
        # TODO: Somehow save coords and orig_shape for each sub_arr.
        # Low priority because this block only used for training data right now.
        if arr[0].ndim == 4 and arr[0].shape[3] == 2:
            arr = [sub_arr[:, :, :, 1] for sub_arr in arr]
        elif arr[0].ndim == 4:
            arr = [sub_arr[:, :, :, 0] for sub_arr in arr]
        arr = [np.expand_dims(sub_arr, axis=3) for sub_arr in arr]

        chunked = [create_chunks(sub_arr, trim=False) for sub_arr in arr]
        arr = np.concatenate([chunk[0] for chunk in chunked])

        # Avoids https://github.com/keras-team/keras/issues/11434
        last_batch_gpus = (arr.shape[0] % batch_size) % num_gpus
        if last_batch_gpus != 0:
            arr = arr[:-last_batch_gpus, :, :, :, :]

        return arr, None, None

    # 2D case with different shapes not implemented
    else:
        raise NotImplementedError()

    # Ensure dimensionality is correct.
    if arr.ndim == 4 and arr.shape[3] == 2:
        arr = arr[:, :, :, 1]
    elif arr.ndim == 4:
        arr = arr[:, :, :, 0]
    arr = np.expand_dims(arr, axis=3)

    # Chunks data
    arr, coords, orig_shape = create_chunks(arr)

    return arr, coords, orig_shape


@ex.capture
def save_test_data(test_path, test_targets_path, test_coords_path,
                   test_shape_path, orig_test_dir, orig_test_targets_dir):
    """Loads, formats, and re-saves test data from original directories."""
    print('in save_test_data')
    # Gets original data files.
    test_files = sorted(absolute_file_paths(orig_test_dir))
    test_targets_files = sorted(absolute_file_paths(orig_test_targets_dir))

    # Loads and preprocesses data.
    test, test_coords, test_shape = load_data(test_files)
    test_targets, _, _ = load_data(test_targets_files, norm=False)

    # Re-saves data in specified directories.
    np.save(test_path, test)
    np.save(test_targets_path, test_targets)
    with open(test_coords_path, "wb") as a, open(test_shape_path, "wb") as b:
        pickle.dump(test_coords, a)
        pickle.dump(test_shape, b)

    return test, test_targets, test_coords, test_shape


@ex.automain
def get_test_data(data_dir):
    """Loads or creates test data."""
    print('in get_test_data')
    os.makedirs(data_dir, exist_ok=True)

    test_path = data_dir + "/test.npy"
    test_targets_path = data_dir + "/test_targets.npy"
    test_coords_path = data_dir + "/test_coords.pickle"
    test_shape_path = data_dir + "/test_shape.pickle"

    try:
        # Loads data if possible.
        test = np.load(test_path)
        test_targets = np.load(test_targets_path)
        with open(test_coords_path, "rb") as a, \
                open(test_shape_path, "rb") as b:
            test_coords = pickle.load(a)
            test_shape = pickle.load(b)
    except (FileNotFoundError, TypeError) as e:
        # Creates data.
        test, test_targets, \
        test_coords, test_shape = save_test_data(test_path,
                                                 test_targets_path,
                                                 test_coords_path,
                                                 test_shape_path)

    input_shape = test[0].shape

    return input_shape, test, test_targets, test_coords, test_shape
