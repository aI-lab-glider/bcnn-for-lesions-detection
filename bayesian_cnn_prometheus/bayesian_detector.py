from typing import Tuple, Dict

import tensorflow as tf

from constants import *
from preprocessing.data_loader import DataLoader


class BayesianDetector:

    def __init__(self, config: Dict):
        preprocessing_config = config.get('preprocessing')
        self.config = config
        self.batch_size = config.get(BATCH_SIZE)
        self.chunk_size = preprocessing_config.get('create_chunks').get(CHUNK_SIZE)
        self._data_loader = DataLoader(preprocessing_config, self.batch_size, self.chunk_size)
        self._train_data: tf.data.Dataset = None
        self._test_data: tf.data.Dataset = None
        self._valid_data: tf.data.Dataset = None
        self._input_shape: Tuple[int] = None
        self._train_len: int = None
        self._model = None

    def load_data(self):
        self._train_data = self._data_loader.get_train()
        self._test_data = self._data_loader.get_test()
        self._valid_data = self._data_loader.get_valid()
        self._input_shape = BayesianDetector.get_input_shape(self._train_data)
        self._train_len = self._calculate_train_len()

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

    def create_model(self):
        print('Getting the model...')
        # self.model, checkpoint_path, kl_alpha = get_model(self.input_shape, scale_factor=self.train_len / self.batch_size,
        #                                              weights_path=weights_path)
        #
        # # Sets callbacks.
        # checkpointer = ModelCheckpoint(checkpoint_path, verbose=1, save_weights_only=True, save_best_only=True)
        #
        # scheduler = LearningRateScheduler(schedule)
        # annealer = Callback() if kl_alpha is None else AnnealingCallback(kl_alpha, kl_start_epoch,
        #                                                                  kl_alpha_increase_per_epoch)

    def fit(self):
        print('Fitting the model...')
        # self.model.fit(x=train_ds, epochs=epochs,
        #             initial_epoch=initial_epoch,
        #             callbacks=[checkpointer, scheduler, annealer],
        #             validation_data=valid_ds,
        #             steps_per_epoch=1, validation_steps=1)
