import os
import numpy as np
import pickle

from .utils import absolute_file_paths, ex, standardize
import tensorflow as tf


@ex.capture
def chunks(arr, batch_size, num_gpus, step, window, trim=True):
    """Chunks a 4D numpy array into smaller 4D arrays."""

    new = []
    coords = []
    shape = arr.shape

    z_max = shape[0] - window[0]
    x_max = shape[1] - window[1]
    y_max = shape[2] - window[2]

    if z_max < 0 or x_max < 0 or y_max < 0:
        raise ValueError("Volume is too small for the given chunk size.")

    z_flag = x_flag = y_flag = False

    # Creates chunks via a sliding rectangular prism window.
    for z in range(0, shape[0], int(window[0] // step)):
        x_flag = y_flag = False

        if z_flag:
            break
        elif z > z_max:
            if z_max == 0 or z_max % int(window[0] // step) == 0:
                break
            z = z_max
            z_flag = True

        for x in range(0, shape[1], int(window[1] // step)):
            y_flag = False

            if x_flag:
                break
            elif x > x_max:
                if x_max == 0 or x_max % int(window[1] // step) == 0:
                    break
                x = x_max
                x_flag = True

            for y in range(0, shape[2], int(window[2] // step)):
                if y_flag:
                    break
                elif y > y_max:
                    if y_max == 0 or y_max % int(window[2] // step) == 0:
                        break
                    y = y_max
                    y_flag = True

                coords.append((z, x, y))
                new.append(arr[z:z + window[0],
                           x:x + window[1],
                           y:y + window[2], :])
    new = np.asarray(new)

    # Avoids https://github.com/keras-team/keras/issues/11434
    if trim:
        last_batch_gpus = (new.shape[0] % batch_size) % num_gpus
        if last_batch_gpus != 0:
            new = new[:-last_batch_gpus, :, :, :, :]
            coords = coords[:-last_batch_gpus]

    return new, coords, shape


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

        chunked = [chunks(sub_arr, trim=False) for sub_arr in arr]
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

    # Chunks data if necessary.
    if vnet:
        arr, coords, orig_shape = chunks(arr)
    else:
        # Avoids https://github.com/keras-team/keras/issues/11434
        last_batch_gpus = (arr.shape[0] % batch_size) % num_gpus
        if last_batch_gpus != 0:
            arr = arr[:-last_batch_gpus, :, :, :]
            coords = None
            orig_shape = arr.shape

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


@ex.capture
def generate_train_data(data_dir: str, batch_size: int):
    train_data_dir = os.path.join(data_dir, 'train')
    train_images_names = os.listdir(train_data_dir)
    batches_num = len(train_images_names) // batch_size
    train_images_names = train_images_names[:batch_size * batches_num]
    train_images_batches = np.array_split(train_images_names, batches_num)

    for image_files_names in train_images_batches:
        images = []
        targets = []

        for image_file_name in image_files_names:
            image_path = os.path.join(train_data_dir, image_file_name)
            target_path = image_path.replace('train', 'train_targets').replace('IMG', 'MASK')

            image = np.load(image_path)
            image = image.reshape((32, 32, 16, 1))
            images.append(image)

            target = np.load(target_path)
            target = target.reshape((32, 32, 16, 1))
            targets.append(target)

        yield np.asarray(images), np.asarray(targets)


@ex.capture
def generate_valid_data(data_dir: str):
    valid_data_dir = os.path.join(data_dir, 'valid')
    for image_file_name in os.listdir(valid_data_dir):
        image_path = os.path.join(valid_data_dir, image_file_name)
        target_path = image_path.replace('valid', 'valid_targets').replace('IMG', 'MASK')

        image = np.load(image_path)
        image = image.reshape((1, 32, 32, 16, 1))

        target = np.load(target_path)
        target = target.reshape((1, 32, 32, 16, 1))

        yield image, target


@ex.capture
def get_train_data(data_dir: str, batch_size: int) -> tuple:
    """
    Loads training and validation data.
    Transforms data into chunks if format is invalid.
    Raises error if data doesn't exist.
    :param data_dir: path to the directory with data
    :param batch_size: size of batch for train data
    :return: input_shape, train_ds, valid_ds
    """
    print('In get_train_data')
    train_ds = tf.data.Dataset.from_generator(generate_train_data, (tf.int64, tf.int64),
                                              (
                                                  tf.TensorShape([16, 32, 32, 16, 1]),
                                                  tf.TensorShape([16, 32, 32, 16, 1])))

    valid_ds = tf.data.Dataset.from_generator(generate_valid_data, (tf.int64, tf.int64),
                                              (tf.TensorShape([1, 32, 32, 16, 1]), tf.TensorShape([1, 32, 32, 16, 1])))
    input_shape = tuple([dim.value for dim in list(train_ds.element_spec[0].shape)])

    return input_shape, train_ds, valid_ds


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
