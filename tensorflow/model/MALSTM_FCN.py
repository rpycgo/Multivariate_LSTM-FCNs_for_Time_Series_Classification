from ..layers.attention import BahdanauAttention
from ..layers.squeeze_and_excitation_network import SqueezeAndExcitationNetworks

import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dropout, Conv1D, BatchNormalization, GlobalAveragePooling1D, RepeatVector, Dense
from tensorflow.keras.models import Model


class MALSTM_FCN(Model):
    def __init__(self, units:int, num_classes: int, reduction_ratio: int):
        super(MALSTM_FCN, self).__init__()
        self.units = units
        self.reduction_ratio = reduction_ratio
        self.num_classes = num_classes

        self.attention = BahdanauAttention(units=self.units)
        self.se_block1 = SqueezeAndExcitationNetworks(reduction_ratio=self.reduction_ratio)
        self.se_block2 = SqueezeAndExcitationNetworks(reduction_ratio=self.reduction_ratio)
    
    def call(self, x: tf.Tensor, attention: bool= True):
        '''
        x: input, N * Q * M
        attention: choose whether to use attention or not
                   if not, use basic lstm

        N: number of samples in dataset
        Q: maximum number of time steps amongst all variables
        M: number of variables
        '''
        # ALSTM-FCN, LSTM-FCN
        dimension_shuffle = tf.transpose(x, [0, 2, 1])  # N, M, Q
        
        if attention:
            lstm = self.attention(dimension_shuffle, dimension_shuffle)[0]  # context_vector, N, M, Q
        else:
            lstm = LSTM(units=x.shape[1], return_sequences=True)(dimension_shuffle) # N, M, Q
        dropout = Dropout(rate=0.3)(lstm)   # N, M, Q

        # Squeeze and Excite
        conv1 = Conv1D(filters=128, kernel_size=8, activation='relu')(x)    # N, Q-8+1, 128
        batch_norm1 = BatchNormalization()(conv1)
        se_block1 = self.se_block1(batch_norm1) # N, Q-8+1, M

        conv2 = Conv1D(filters=256, kernel_size=5, activation='relu')(se_block1)    # N, Q-8+1-5+1, 256
        batch_norm2 = BatchNormalization()(conv2)
        se_block2 = self.se_block2(batch_norm2) # N, Q-8+1-5+1, 256

        conv3 = Conv1D(filters=128, kernel_size=3, activation='relu')(se_block2)    # N, Q-8+1-5+1-3+1, 128
        batch_norm3 = BatchNormalization()(conv3)

        global_pooling = GlobalAveragePooling1D()(batch_norm3)  # N, 128
        global_pooling_for_concat = RepeatVector(dropout.shape[1])(global_pooling)  # N, M, 128

        concat = tf.concat([dropout, global_pooling_for_concat], axis=-1)   # N, M, 128+Q
        output = Dense(units=self.num_classes)(concat)

        return tf.nn.softmax(output, axis=-1) 
