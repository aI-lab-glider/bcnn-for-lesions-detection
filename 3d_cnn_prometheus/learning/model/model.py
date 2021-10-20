import os

import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam

from .bayesian_vnet import bayesian_vnet
from .utils import ex, variational_free_energy_loss, get_latest_file


@ex.capture
def load_model(input_shape, weights_path, net, prior_std,
               kernel_size, activation, padding):
    """Loads model from .h5 file.

    If model is saved as multi-gpu, re-saves it as single-gpu.
    """

    # Loads single-gpu model.
    model = net(input_shape,
                kernel_size=kernel_size,
                activation=activation,
                padding=padding,
                prior_std=prior_std)
    model.load_weights(weights_path)

    return model


@ex.capture
def get_paths(network_type: str, weights_path, weights_dir):
    checkpoint_path = (weights_dir + f"/{network_type}/{network_type}" + "-{epoch:02d}"
                                                                         "-{val_acc:.3f}-{val_loss:.0f}.h5")
    if not weights_path:
        weights_path = get_latest_file(weights_dir + f"/{network_type}")

    if not os.path.isfile(weights_path):
        raise Exception(f'Dir with weights {weights_path} does not exist')

    return checkpoint_path, weights_path


@ex.capture
def get_model(input_shape: tuple, weights_dir: str, resume: bool, prior_std: float, kernel_size: int, activation: str,
              padding: int, kl_alpha: float, kl_start_epoch: int, kl_alpha_increase_per_epoch: float,
              initial_epoch: int, scale_factor: int = 1):
    """
    Loads or creates model. If a weights path is specified, loads from that path.
    Otherwise, loads the most recently modified model.
    :param input_shape: input batch shape (tuple)
    :param weights_dir: path to the dir with weights or where weights should be saved (str)
    :param resume: start training from the last checkpoint (bool)
    :param prior_std: prior distribution standard deviation, a param for normal distribution (float)
    :param kernel_size: kernel size of conv filters in network (int)
    :param activation: activation function (str)
    :param padding: padding for conv filters (int)
    :param kl_alpha: an initial value the KL weight (article: k_0) (float)
    :param kl_start_epoch: epoch at which to start increasing KL weight (article: s) (int)
    :param kl_alpha_increase_per_epoch: step value to obtain the KL weight for the current epoch (article: k_1) (float)
    :param initial_epoch: epoch at which to start training (useful for resuming a previous training run) (int)
    :param scale_factor: num_samples / batch_size, (article: M) (float)
    :return:
    """
    os.makedirs(weights_dir + "/bayesian", exist_ok=True)

    # Sets variables for bayesian model.
    checkpoint_path, latest_weights_path = get_paths('bayesian')

    # Loads or creates model.
    if latest_weights_path and resume:
        model = load_model(input_shape, latest_weights_path, bayesian_vnet)
    else:
        model = bayesian_vnet(input_shape,
                              kernel_size=kernel_size,
                              activation=activation,
                              padding=padding,
                              prior_std=prior_std)

    # Prints model summary.
    model.summary(line_length=127)

    # Sets loss function.
    if initial_epoch >= kl_start_epoch:
        kl_alpha = min(1., kl_alpha + (initial_epoch - kl_start_epoch) * kl_alpha_increase_per_epoch)

    kl_alpha = K.variable(kl_alpha)
    loss = variational_free_energy_loss(model, scale_factor, kl_alpha)

    # Compiles model with Adam optimizer.
    model.compile(loss=loss,
                  optimizer=Adam(),
                  metrics=["accuracy"])

    return model, checkpoint_path, kl_alpha
