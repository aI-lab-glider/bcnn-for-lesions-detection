import math
import os

from tensorflow.keras.callbacks import Callback, LearningRateScheduler, ModelCheckpoint
import tensorflow as tf

from model.dataset import get_train_data
from model.model import get_model
from model.utils import AnnealingCallback, ex

# Ignores TensorFlow CPU messages.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def generate_train_data():
    import numpy as np
    train_data_dir = '/home/alikrz/Pulpit/MyStuff/cancer-detection/3d-cnn-prometheus/data/chunks/train'
    train_images_names = os.listdir(train_data_dir)
    batches_num = len(train_images_names) // 16
    train_images_names = train_images_names[:16*batches_num]

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


def generate_valid_data():
    import numpy as np
    train_data_dir = '/home/alikrz/Pulpit/MyStuff/cancer-detection/3d-cnn-prometheus/data/chunks/valid'
    for image_file_name in os.listdir(train_data_dir):
        image_path = os.path.join(train_data_dir, image_file_name)
        target_path = image_path.replace('valid', 'valid_targets').replace('IMG', 'MASK')

        image = np.load(image_path)
        image = image.reshape((1, 32, 32, 16, 1))

        target = np.load(target_path)
        target = target.reshape((1, 32, 32, 16, 1))

        yield image, target


@ex.capture
def schedule(epoch: int, initial_learning_rate: float, lr_decay_start_epoch: int) -> float:
    """
    Defines exponentially decaying learning rate.
    :param epoch: actual epoch number (int)
    :param initial_learning_rate: initial learning rate (float)
    :param lr_decay_start_epoch: epoch number since the learning rate is decaying (int)
    :return: updated value of the learning rate (float)
    """
    if epoch < lr_decay_start_epoch:
        return initial_learning_rate
    else:
        return initial_learning_rate * math.exp((10 * initial_learning_rate) * (lr_decay_start_epoch - epoch))


@ex.automain
def train(weights_path: str, epochs: int, batch_size: int, initial_epoch: int, kl_start_epoch: int,
          kl_alpha_increase_per_epoch: float) -> None:
    """
    Trains a model.
    :param weights_path: path to save updated weights (or path to trained before weights if they exist) (str)
    :param epochs: number of epochs to train the model (int)
    :param batch_size: number of samples in one batch (samples participating in training during one epoch) (int)
    :param initial_epoch: epoch at which to start training (useful for resuming a previous training run) (int)
    :param kl_start_epoch: epoch at which to start increasing KL weight (article: s) (int)
    :param kl_alpha_increase_per_epoch: step value to obtain the KL weight for the current epoch (article: k_1) (float)
    """
    print('Loading data...')
    # Loads or creates training data.
    # input_shape, train, valid, train_targets, valid_targets = get_train_data()
    # train_len = len(train)

    train_ds = tf.data.Dataset.from_generator(generate_train_data, (tf.int64, tf.int64),
                                              (tf.TensorShape([16, 32, 32, 16, 1]), tf.TensorShape([16, 32, 32, 16, 1])))

    valid_ds = tf.data.Dataset.from_generator(generate_valid_data, (tf.int64, tf.int64),
                                              (tf.TensorShape([1, 32, 32, 16, 1]), tf.TensorShape([1, 32, 32, 16, 1])))

    gen = generate_train_data()
    input_shape = next(gen)[0][0].shape
    train_len = len(os.listdir('/home/alikrz/Pulpit/MyStuff/cancer-detection/3d-cnn-prometheus/data/chunks/train'))
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

    # model.fit(train, train_targets, batch_size, epochs,
    #           initial_epoch=initial_epoch,
    #           callbacks=[checkpointer, scheduler, annealer],
    #           validation_data=(valid, valid_targets))


if __name__ == '__main__':
    train()
