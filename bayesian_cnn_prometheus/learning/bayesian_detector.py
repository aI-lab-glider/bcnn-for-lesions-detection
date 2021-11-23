from pathlib import Path
from typing import Tuple, Dict

import math
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint
from tensorflow_core.python.keras.callbacks import Callback
from tensorflow_core.python.keras.optimizer_v2.adam import Adam

from .model.bayesian_vnet import bayesian_vnet
from .model.utils import AnnealingCallback, variational_free_energy_loss


class BayesianDetector:

    def __init__(self, config: Dict, batch_size: int, input_shape: Tuple[int, ...]):
        self._config = config
        self._input_shape: Tuple[int, ...] = None
        self._train_len: int = None
        self._model = None
        self._batch_size = batch_size

        self._train_data = None
        self._valid_data = None
        self._test_data = None

        # For creating a model
        self._kernel_size = config.get('kernel_size')
        self._activation = config.get('activation')
        self._padding = config.get('padding')
        self._prior_std = config.get('prior_std')

        self._initial_epoch = config.get('initial_epoch')
        self._kl_start_epoch = config.get('kl_start_epoch')
        self._kl_alpha_increase_per_epoch = config.get('kl_alpha_increase_per_epoch')
        self._kl_alpha = self._adjust_kl_alpha(config.get('kl_alpha'))

        self._weights_path = config.get('weights_path')
        self._weights_dir = config.get('weights_dir')
        self._checkpointer: ModelCheckpoint = None
        self._scheduler: LearningRateScheduler = None
        self._annealer: Callback = None
        self._kl_start_epoch: int = config.get('kl_start_epoch')
        self._kl_alpha_increase_per_epoch: float = config.get('kl_alpha_increase_per_epoch')

        # For fitting
        self._lr_decay_start_epoch = config.get('lr_decay_start_epoch')
        self._epochs = config.get('epochs')
        self._validation_steps = config.get('validation_steps')
        self._initial_epoch = config.get('initial_epoch')
        self._valid_ds = config.get('valid_ds')

        self._initialize_model(input_shape)
        self._initialize_callbacks()

    def _initialize_model(self, input_shape: Tuple[int, ...]):
        train_len = self._calculate_train_len()

        self._model = bayesian_vnet(input_shape, kernel_size=self._kernel_size, activation=self._activation,
                                    padding=self._padding, prior_std=self._prior_std)
        self._model.summary(line_length=127)
        loss_function = variational_free_energy_loss(self._model, train_len / self._batch_size, self._kl_alpha)
        self._model.compile(loss=loss_function, optimizer=Adam(), metrics=["accuracy"])

    def _initialize_callbacks(self):
        self._checkpoint_path = BayesianDetector._get_paths('bayesian', self._weights_dir)
        self._checkpointer = ModelCheckpoint(str(self._checkpoint_path), verbose=1, save_weights_only=True,
                                             save_best_only=True, )
        self._scheduler = LearningRateScheduler(BayesianDetector._get_scheduler(self._lr_decay_start_epoch))
        self._annealer = Callback() if self._kl_alpha is None else \
            AnnealingCallback(self._kl_alpha, self._kl_start_epoch, self._kl_alpha_increase_per_epoch)

    def _adjust_kl_alpha(self, kl_alpha: int):
        if kl_alpha is None:
            kl_alpha = 0.2  # TODO create a place to store default values// .2 is just a blind guess
        if self._initial_epoch >= self._kl_start_epoch:
            kl_alpha = min(1.,
                           kl_alpha + (self._initial_epoch - self._kl_start_epoch) * self._kl_alpha_increase_per_epoch)

        return K.variable(kl_alpha)

    def fit(self, training_dataset, validation_dataset):
        print('Fitting the model...')
        self._model.fit(x=training_dataset, epochs=self._epochs, initial_epoch=self._initial_epoch,
                        callbacks=[self._checkpointer, self._scheduler, self._annealer],
                        validation_data=validation_dataset.repeat(), validation_steps=self._validation_steps)

    @staticmethod
    def _get_paths(network_type: str, weights_dir: Path):
        Path(weights_dir).mkdir(parents=True, exist_ok=True)
        checkpoint_path = Path(weights_dir) / (network_type + "-{epoch:02d}-{val_acc:.3f}-{val_loss:.0f}.h5")
        return checkpoint_path

    @staticmethod
    def _get_scheduler(lr_decay_start_epoch: int) -> float:
        """
        Defines exponentially decaying learning rate.
        :param lr_decay_start_epoch: epoch number since the learning rate is decaying
        :return: updated value of the learning rate
        """

        def schedule(epoch: int, initial_learning_rate: float):
            if epoch >= lr_decay_start_epoch:
                initial_learning_rate *= math.exp((10 * initial_learning_rate) * (lr_decay_start_epoch - epoch))
            return initial_learning_rate

        return schedule

    @staticmethod
    def get_input_shape(dataset: tf.data.Dataset) -> Tuple[int, ...]:
        """
        Get input dataset from tensorflow dataset object.
        :param dataset: tensorflow dataset object
        :return: input shape
        """
        return tuple([int(dim.value) for dim in list(dataset.element_spec[0].shape)[1:]])

    @staticmethod
    def _calculate_train_len():
        return 100  # TODO How can we calc train len?
