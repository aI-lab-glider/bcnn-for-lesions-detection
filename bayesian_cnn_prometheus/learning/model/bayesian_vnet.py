from abc import ABC

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, UpSampling3D, concatenate
from tensorflow.keras.models import Model
from bayesian_cnn_prometheus.learning.model.groupnorm import GroupNormalization


def down_stage(inputs, filters, kernel_size=3,
               activation="relu", padding="SAME"):
    conv = Conv3D(filters, kernel_size,
                  activation=activation, padding=padding)(inputs)
    conv = GroupNormalization()(conv)
    conv = Conv3D(filters, kernel_size,
                  activation=activation, padding=padding)(conv)
    conv = GroupNormalization()(conv)
    pool = MaxPooling3D()(conv)
    return conv, pool


def up_stage(inputs, skip, filters, prior_fn, kernel_divergence_fn, kernel_size=3, activation="relu", padding="SAME"):
    up = UpSampling3D()(inputs)
    up = tfp.layers.Convolution3DFlipout(filters, 2,
                                         activation=activation,
                                         padding=padding,
                                         kernel_prior_fn=prior_fn,
                                         kernel_divergence_fn=kernel_divergence_fn)(up)
    up = GroupNormalization()(up)

    merge = concatenate([skip, up])
    merge = GroupNormalization()(merge)

    conv = tfp.layers.Convolution3DFlipout(filters, kernel_size,
                                           activation=activation,
                                           padding=padding,
                                           kernel_prior_fn=prior_fn,
                                           kernel_divergence_fn=kernel_divergence_fn)(merge)
    conv = GroupNormalization()(conv)
    conv = tfp.layers.Convolution3DFlipout(filters, kernel_size,
                                           activation=activation,
                                           padding=padding,
                                           kernel_prior_fn=prior_fn,
                                           kernel_divergence_fn=kernel_divergence_fn)(conv)
    conv = GroupNormalization()(conv)

    return conv


def end_stage(inputs, prior_fn, kernel_divergence_fn, kernel_size=3, activation="relu", padding="SAME"):
    conv1 = tfp.layers.Convolution3DFlipout(1, kernel_size,
                                            activation=activation,
                                            padding=padding,
                                            kernel_prior_fn=prior_fn,
                                            kernel_divergence_fn=kernel_divergence_fn)(inputs)
    conv = tfp.layers.Convolution3DFlipout(1, 1, activation="sigmoid",
                                           kernel_prior_fn=prior_fn,
                                           kernel_divergence_fn=kernel_divergence_fn)(conv1)

    return conv


class BayesianVnet(Model, ABC):
    def __init__(self, input_shape, kernel_size=3, activation="relu", padding="SAME", **kwargs):
        prior_std = kwargs.get("prior_std", 1)
        prior_fn = normal_prior(prior_std)
        kernel_divergence_fn = lambda q, p, _: tfp.distributions.kl_divergence(q, p) / (input_shape[0] * 1.0)

        inputs = Input(input_shape)
        conv1, pool1 = down_stage(inputs, 16,
                                  kernel_size=kernel_size,
                                  activation=activation,
                                  padding=padding)
        conv2, pool2 = down_stage(pool1, 32,
                                  kernel_size=kernel_size,
                                  activation=activation,
                                  padding=padding)
        conv3, pool3 = down_stage(pool2, 64,
                                  kernel_size=kernel_size,
                                  activation=activation,
                                  padding=padding)
        conv4, _ = down_stage(pool3, 128,
                              kernel_size=kernel_size,
                              activation=activation,
                              padding=padding)

        conv5 = up_stage(conv4, conv3, 64, prior_fn,
                         kernel_size=kernel_size,
                         activation=activation,
                         padding=padding,
                         kernel_divergence_fn=kernel_divergence_fn)
        conv6 = up_stage(conv5, conv2, 32, prior_fn,
                         kernel_size=kernel_size,
                         activation=activation,
                         padding=padding,
                         kernel_divergence_fn=kernel_divergence_fn)
        conv7 = up_stage(conv6, conv1, 16, prior_fn,
                         kernel_size=kernel_size,
                         activation=activation,
                         padding=padding,
                         kernel_divergence_fn=kernel_divergence_fn)

        conv8 = end_stage(conv7, prior_fn,
                          kernel_size=kernel_size,
                          activation=activation,
                          padding=padding,
                          kernel_divergence_fn=kernel_divergence_fn)

        super(BayesianVnet, self).__init__(inputs=inputs, outputs=conv8)


def normal_prior(prior_std):
    """Defines normal distribution prior for Bayesian neural network."""

    def prior_fn(dtype, shape, name, trainable, add_variable_fn):
        tfd = tfp.distributions
        dist = tfd.Normal(loc=tf.zeros(shape, dtype),
                          scale=dtype.as_numpy_dtype((prior_std)))
        batch_ndims = tf.size(input=dist.batch_shape_tensor())
        return tfd.Independent(dist, reinterpreted_batch_ndims=batch_ndims)

    return prior_fn
