import tensorflow as tf
from tensorflow.keras.layers import Layer, GlobalAveragePooling1D, Dense, Conv1D, RepeatVector


class SqueezeAndExcitationNetworks(Layer):
    def __init__(self, reduction_ratio, **kwargs):
        super(SqueezeAndExcitationNetworks, self).__init__(**kwargs)
        self.reduction_ratio = reduction_ratio

    def call(self, x):
        filters=x.shape[-1]

        u = Conv1D(filters=filters, kernel_size=1)(x)   # batch_size, time_seq, filters
        average_pooling = GlobalAveragePooling1D()(u)   # batch_size, filters
        fc1 = Dense(units=(filters // self.reduction_ratio), activation='relu')(average_pooling)    # batch_size, filters
        fc2 = Dense(units=filters, activation='sigmoid')(fc1)   # batch_size, filters

        fc2_repeated = RepeatVector(x.shape[1])(fc2)    # batch_size, time_seq, filters

        return u * fc2_repeated
