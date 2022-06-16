import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Conv1D


class SqueezeAndExcitationNetworks(Layer):
    def __init__(self, reduction_ratio, **kwargs):
        super(SqueezeAndExcitationNetworks, self).__init__(**kwargs)
        self.reduction_ratio = reduction_ratio

    def call(self, x):
        filters=x.shape[-1]

        u = Conv1D(filters=filters, kernel_size=1)(x)   # batch_size, time_seq, filters
        _global_average_pooling = tf.reduce_mean(u, axis=(0, 1))   # filters
        global_average_pooling = _global_average_pooling[tf.newaxis, tf.newaxis, :]   # 1, 1, filters
        fc1 = Dense(units=(filters // self.reduction_ratio), activation='relu')(global_average_pooling)    # 1, 1, filters/reduction_ratio
        fc2 = Dense(units=filters, activation='sigmoid')(fc1)   # 1, 1, filters

        return u * fc2
