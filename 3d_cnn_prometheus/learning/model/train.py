import math
import os

from tensorflow.keras.callbacks import Callback, LearningRateScheduler, ModelCheckpoint

from .dataset import get_train_dataset, get_valid_dataset, get_input_shape
from .experiment_setup import ex
from .model import get_model
from .utils import AnnealingCallback
from .constants import *

# Ignores TensorFlow CPU messages.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


@ex.capture
def schedule(epoch: int, initial_learning_rate: float, lr_decay_start_epoch: int) -> float:
    """
    Defines exponentially decaying learning rate.
    :param epoch: actual epoch number
    :param initial_learning_rate: initial learning rate
    :param lr_decay_start_epoch: epoch number since the learning rate is decaying
    :return: updated value of the learning rate
    """
    if epoch < lr_decay_start_epoch:
        return initial_learning_rate
    else:
        return initial_learning_rate * math.exp((10 * initial_learning_rate) * (lr_decay_start_epoch - epoch))


@ex.capture
def train(weights_path: str, epochs: int, batch_size: int, initial_epoch: int, kl_start_epoch: int,
          kl_alpha_increase_per_epoch: float) -> None:
    """
    Trains a model.
    :param weights_path: path to save updated weights (or path to trained before weights if they exist)
    :param epochs: number of epochs to train the model
    :param batch_size: number of samples in one batch (samples participating in training during one epoch)
    :param initial_epoch: epoch at which to start training (useful for resuming a previous training run)
    :param kl_start_epoch: epoch at which to start increasing KL weight (article: s)
    :param kl_alpha_increase_per_epoch: step value to obtain the KL weight for the current epoch (article: k_1)
    """
    print('Loading data...')
    train_ds = get_train_dataset()
    valid_ds = get_valid_dataset()
    input_shape = get_input_shape(train_ds)

    train_len = len(os.listdir(CHUNKS_TRAIN_PATH))

    print('Getting the model...')
    # Loads or creates model.
    model, checkpoint_path, kl_alpha = get_model(input_shape, scale_factor=train_len / batch_size,
                                                 weights_path=weights_path)

    # Sets callbacks.
    checkpointer = ModelCheckpoint(checkpoint_path, verbose=1, save_weights_only=True, save_best_only=True)

    scheduler = LearningRateScheduler(schedule)
    annealer = Callback() if kl_alpha is None else AnnealingCallback(kl_alpha, kl_start_epoch,
                                                                     kl_alpha_increase_per_epoch)

    print('Fitting the model...')
    # Trains model.
    model.fit(x=train_ds, epochs=epochs,
              initial_epoch=initial_epoch,
              callbacks=[checkpointer, scheduler, annealer],
              validation_data=valid_ds,
              steps_per_epoch=1, validation_steps=1)
