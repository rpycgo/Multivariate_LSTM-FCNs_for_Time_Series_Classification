import tensorflow as tf
from tensorflow.keras.layers import Dense, Layer


class BahdanauAttention(Layer):
  def __init__(self, units, **kwargs):
    super(BahdanauAttention, self).__init__(**kwargs)
    self.W1 = Dense(units)
    self.W2 = Dense(units)
    self.V = Dense(1)

  def call(self, query, values):
    '''
    query: previous output (batch_size, hidden size)
    values: last hidden states of each time in the encoders (batch_size, T, hidden_size)
    '''    
    query_with_time_axis = tf.expand_dims(query, 1) # (batch_size, 1, hidden_size)
    
    _score = tf.nn.tanh(self.W1(query_with_time_axis) + self.W2(values)) # (batch_size, T, units)
    score = self.V(_score)   # (batch_size, T, 1)
    
    attention_weights = tf.nn.softmax(score, axis=1)    #(batch_size, T, 1)

    context_vector = attention_weights * values    #(batch_size, T, 1)
    context_vector = tf.reduce_sum(context_vector, axis=1)  # (batch_size, hidden_size)

    return context_vector, attention_weights
