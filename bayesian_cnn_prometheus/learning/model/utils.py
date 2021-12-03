import os

import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.python.keras.losses import kl_divergence


def round_down(num, factor):
    """Rounds num to next lowest multiple of factor."""

    return (num // factor) * factor


def acc(a, b):
    """Calculates number of matches in two np arrays."""
    return np.count_nonzero(a == b) / a.size


def absolute_file_paths(directory, match=""):
    """Gets absolute file paths from a directory.
    Does not include subdirectories.
    Args:
        match: Returns only paths of files containing the given string.
    """
    paths = []
    for root, dirs, filenames in os.walk(directory):
        for f in filenames:
            if match in f:
                paths.append(os.path.abspath(os.path.join(root, f)))
        break
    return paths


def standardize(raw):
    """Transforms data to have mean 0 and std 1."""

    return (raw - np.mean(raw)) / np.std(raw)


def variational_free_energy_loss(kl_alpha):
    """
    Defines variational free energy loss.
    Sum of KL divergence (supplied by tfp) and binary cross-entropy.
    """
    def loss(y_true, y_pred):
        bce = binary_crossentropy(y_true, y_pred)
        kl = kl_divergence(y_true, y_pred)
        return bce + K.get_value(kl_alpha) * kl

    return loss


def get_latest_file(directory, match=""):
    """Gets the absolute file path of the last modified file in a directory.
    Args:
        match: Returns only paths of files containing the given string.
    """

    paths = absolute_file_paths(directory, match=match)
    if paths:
        return max(paths, key=os.path.getctime)
    else:
        return None


class AnnealingCallback(Callback):
    def __init__(self, kl_alpha, kl_start_epoch, kl_alpha_increase_per_epoch):
        self.kl_alpha = kl_alpha
        self.kl_start_epoch = kl_start_epoch
        self.kl_alpha_increase_per_epoch = kl_alpha_increase_per_epoch

    def on_epoch_end(self, epoch, logs={}):
        if epoch >= self.kl_start_epoch - 2:
            new_kl_alpha = min(K.get_value(self.kl_alpha) + self.kl_alpha_increase_per_epoch, 1.)
            K.set_value(self.kl_alpha, new_kl_alpha)
        print("Current KL Weight is " + str(K.get_value(self.kl_alpha)))