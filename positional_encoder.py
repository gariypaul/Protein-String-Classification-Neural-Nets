'''

Positional Encoder Layer

This layer takes as input a tensor that represents a batch of a sequence of 
encoded tokens (examplex x seq len x token embedding), and combines it with
an encoding of the position of each of the tokens.

For the default combination_type of 'add', the positional encoding vectors
are the same size as the token encoding vectors; the two are added together.

Source: Hands-On Machine Learning, p 558

'''
import numpy as np
import tensorflow as tf
from tensorflow import keras

class PositionalEncoding(keras.layers.Layer):
    def __init__(self, max_steps:int, max_dims:int, combination_type:str='add',
                 dtype=tf.float32, **kwargs):
        '''
        Constructor

        :param max_steps: the number of tokens in the sequence
        :param max_dims: the length of the vector used to encode position
                    (must match the token encoding length if "add")
        :param combination_type: "add" works right now
        :param dtype: The type used for encoding of position
        '''
        # Call superclass constructor
        super().__init__(dtype=dtype, **kwargs)

        # Deal with odd lengths
        if max_dims % 2 == 1: max_dims += 1

        # Create the positional representation
        p, i = np.meshgrid(np.arange(max_steps), np.arange(max_dims // 2))
        pos_emb = np.empty((1, max_steps, max_dims))
        pos_emb[0, :, ::2] = np.sin(p / 10000**(2 * i / max_dims)).T
        pos_emb[0, :, 1::2] = np.cos(p / 10000**(2 * i / max_dims)).T

        # Save the state
        self.positional_embedding = tf.constant(pos_emb.astype(self.dtype))
        self.combination_type = combination_type
        
    def call(self, inputs):
        '''
        This method is what implements the object "callable" property.

        Determines how the input tensor is translated into the output tensor.

        :param inputs: TF Tensor
        :return: TF Tensor
        '''
        shape = tf.shape(inputs)
        embed = self.positional_embedding[:, :shape[-2], :shape[-1]]
        if self.combination_type == 'add':
            return inputs + embed
        elif self.combination_type == 'concat':
            # Not clear why gradient tape doesn't like this
            return tf.concat([inputs, embed], axis=2)

    def embedding(self):
        return self.positional_embedding
