from typing import Tuple, Dict

import math
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint
from tensorflow_core.python.keras.optimizer_v2.adam import Adam

from network.utils import *
from .model.bayesian_vnet import bayesian_vnet


class BayesianDetector:

    def __init__(self, config: Dict):
        preprocessing_config = config.get('../preprocessing')
        self.config = config
        self._input_shape: Tuple[int] = None
        self._train_len: int = None
        self._model = None

        self._train = None
        self._valid = None
        self._test = None

        # For creating a model
        self._kernel_size = config.get('kernel_size')
        self._activation = config.get('activation')
        self._padding = config.get('padding')
        self._prior_std = config.get('prior_std')

        self._initial_epoch = config.get('initial_epoch')
        self._kl_start_epoch = config.get('kl_start_epoch')
        self._kl_alpha_increase_per_epoch = config.get('kl_alpha_increase_per_epoch')
        self._kl_alpha = self._adjust_kl_alpha(config.get('kl_alpha'))
        self._scale_factor = config.get('scale_factor')

        self._weights_path = config.get('weights_path')
        self._weights_dir = config.get('weights_dir')
        self._checkpointer: ModelCheckpoint = None
        self._scheduler: LearningRateScheduler = None
        self._annealer: Callback = None
        self._kl_start_epoch: int = config.get('kl_start_epoch')
        self._kl_alpha_increase_per_epoch: float = config.get('kl_alpha_increase_per_epoch')

        # For fitting
        self._train_ds = config.get('train_ds')
        self._epochs = config.get('epochs')
        self._initial_epoch = config.get('initial_epoch')
        self._valid_ds = config.get('valid_ds')

    def _adjust_kl_alpha(self, kl_alpha: int):
        if self._initial_epoch >= self._kl_start_epoch:
            kl_alpha = min(1.,
                           kl_alpha + (self._initial_epoch - self._kl_start_epoch) * self._kl_alpha_increase_per_epoch)

        return K.variable(kl_alpha)

    def put_train_data(self, train_data):  # TODO think if is it the right way
        self._train = train_data
        self._input_shape = BayesianDetector.get_input_shape(train_data)

    def create_model(self):
        self._initialize_model()
        self._initialize_callbacks()
        # self._input_shape = DataLoader.get_input_shape(self._train_data)
        # self._train_len = self._calculate_train_len()

    def _initialize_model(self):
        self._model = bayesian_vnet(self._input_shape, kernel_size=self._kernel_size, activation=self._activation,
                                    padding=self._padding, prior_std=self._prior_std)
        self._model.summary(line_length=127)
        loss_function = variational_free_energy_loss(self._model, self._scale_factor, self._kl_alpha)
        self._model.compile(loss=loss_function, optimizer=Adam(), metrics=["accuracy"])

    def _initialize_callbacks(self):
        self._checkpoint_path = BayesianDetector.get_paths('bayesian', self._weights_dir)  # TODO ProxPxD Refactor
        self._checkpointer = ModelCheckpoint(self._checkpoint_path, verbose=1, save_weights_only=True,
                                             save_best_only=True, )
        self._scheduler = LearningRateScheduler(BayesianDetector._schedule)
        self._annealer = Callback() if self._kl_alpha is None else AnnealingCallback(self._kl_alpha,
                                                                                     self._kl_start_epoch,
                                                                                     self._kl_alpha_increase_per_epoch)

    @staticmethod
    def get_paths(network_type: str, weights_dir):
        checkpoint_path = (weights_dir + f"/{network_type}/{network_type}" + "-{epoch:02d}"
                                                                             "-{val_acc:.3f}-{val_loss:.0f}.h5")
        return checkpoint_path

    @staticmethod
    def _schedule(epoch: int, initial_learning_rate: float, lr_decay_start_epoch: int) -> float:
        """
        Defines exponentially decaying learning rate.
        :param epoch: actual epoch number
        :param initial_learning_rate: initial learning rate
        :param lr_decay_start_epoch: epoch number since the learning rate is decaying
        :return: updated value of the learning rate
        """
        if lr_decay_start_epoch <= epoch:
            initial_learning_rate *= math.exp((10 * initial_learning_rate) * (lr_decay_start_epoch - epoch))
        return initial_learning_rate

    def fit(self):
        print('Fitting the model...')
        self._model.fit(x=self._train_ds, epochs=self._epochs,
                        initial_epoch=self._initial_epoch,
                        callbacks=[self._checkpointer, self._scheduler, self._annealer],
                        validation_data=self._valid_ds,
                        steps_per_epoch=1, validation_steps=1)

    @staticmethod
    def get_input_shape(dataset: tf.data.Dataset) -> Tuple[int]:
        """
        Get input dataset from tensorflow dataset object.
        :param dataset: tensorflow dataset object
        :return: input shape
        """
        return tuple([int(dim.value) for dim in list(dataset.element_spec[0].shape)])

    @staticmethod
    def _calculate_train_len():
        return 100  # TODO How can we calc train len?
