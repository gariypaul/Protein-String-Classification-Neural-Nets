"""
Main script for building the neural network model.
This script defines a function to build a neural network model using Keras and TensorFlow.

Author: Paul Gariy
Date: 2025-05-22
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import (
    SimpleRNN,
    GRU,
    MultiHeadAttention,
    Dense,
    Dropout,
    Embedding,
    BatchNormalization,
    Conv1D,
    AveragePooling1D,
    GlobalMaxPooling1D,
)
from tensorflow.keras import layers, regularizers
from tensorflow.keras import Model, Input
from positional_encoder import (
    PositionalEncoding,
)


def build_network(
    len_max: int,
    n_tokens: int,
    nhidden_layers: list[int],
    ndense_layers: list[int],
    dense_activations: list[str],
    dropout: float = 0.2,
    L1: float | None = None,
    L2: float | None = None,
    recurrent_L1: float | None = None,
    recurrent_L2: float | None = None,
    activation: str = "relu",
    num_classes: int = 46,
    embedding_dimension: int = 128,
    network_type: str = "RNN",
    num_heads: int = 8,
    batch_normalization: bool = False,
    key_dim: int = 18,
):
    """
    Build the network model graph. The network will depend on the network_type and the args.
    For RNN and GRU, the network will be a stack of RNN or GRU layers with the specified number of units.
    For ATTN, the network will be a stack of MultiHeadAttention layers with the specified number of heads.
    The network will also include a dense layer with the specified number of units and activation function.

    Parameters:
    len_max (int): The maximum length of the input sequences.
    n_tokens (int): The number of unique tokens in the vocabulary.
    nhidden_layers (list[int]): A list of integers representing the number of units in each hidden layer.
    dropout (float): The dropout rate to apply after each hidden layer.
    L1 (float | None): L1 regularization strength. If None, no L1 regularization is applied.
    L2 (float | None): L2 regularization strength. If None, no L2 regularization is applied.
    recurrent_L1 (float | None): L1 regularization strength for the recurrent layers. If None, no L1 regularization is applied.
    recurrent_L2 (float | None): L2 regularization strength for the recurrent layers. If None, no L2 regularization is applied.
    activation (str): The activation function to use in the hidden layers. Default is "relu".
    num_heads (int): The number of attention heads for the MultiHeadAttention layer. Default is 8.
    batch_normalization (bool): Whether to apply batch normalization after each hidden layer. Default is False.
    num_classes (int): The number of output classes. Default is 46.
    embedding_dim (int): The dimension of the embedding layer. Default is 128.
    network_type (str): The type of network to build. Options are "RNN", "GRU", or "ATTN".
    """
    # Select kernel regularization based on L1 and L2 values
    kernel_regularizer = None
    if L1 is not None:
        kernel_regularizer = regularizers.l1(L1)
        print(f"L1 regularization: {L1}")
    if L2 is not None:
        kernel_regularizer = regularizers.l2(L2)
        print(f"L2 regularization: {L2}")
    if L1 is not None and L2 is not None:
        kernel_regularizer = regularizers.l1_l2(L1, L2)
        print(f"L1 and L2 regularization: {L1}, {L2}")

    # Select recurrent kernel regularization based on recurrent_regularizer and recurrent_regularizer_strength
    recurrent_regularizer = None
    if recurrent_L1 is not None:
        recurrent_regularizer = regularizers.l1(recurrent_L1)
        print(f"Recurrent L1 regularization: {recurrent_L1}")
    if recurrent_L2 is not None:
        recurrent_regularizer = regularizers.l2(recurrent_L2)
        print(f"Recurrent L2 regularization: {recurrent_L2}")
    if recurrent_L1 is not None and recurrent_L2 is not None:
        recurrent_regularizer = regularizers.l1_l2(recurrent_L1, recurrent_L2)
        print(f"Recurrent L1 and L2 regularization: {recurrent_L1}, {recurrent_L2}")

    # Create the input layer
    tensor = input_tensor = Input(shape=(3934,))
    # Create the embedding layer
    tensor = Embedding(
        input_dim=24, output_dim=embedding_dimension, input_length=3934
    )(tensor)

    # if multihead attention, add positional encoding
    if network_type == "ATTN":
        tensor = PositionalEncoding(3934, embedding_dimension)(tensor)

    # Preprocessing Convolutional Layer with striding to reduce the length of the string
    tensor = Conv1D(
        filters=16, kernel_size=3, strides=2, padding="same", activation="relu"
    )(tensor)

    for i, units in enumerate(nhidden_layers):
        last_recurr = i == len(nhidden_layers) - 1
        if network_type == "RNN":
            tensor = SimpleRNN(
                units,
                return_sequences=not last_recurr,
                kernel_regularizer=kernel_regularizer,
                recurrent_regularizer=recurrent_regularizer,
                activation=activation,
            )(tensor)
        elif network_type == "GRU":
            tensor = GRU(
                units,
                return_sequences=not last_recurr,
                kernel_regularizer=kernel_regularizer,
                recurrent_regularizer=recurrent_regularizer,
                activation=activation,
            )(tensor)
        elif network_type == "ATTN":
            attention = MultiHeadAttention(
                num_heads=8,
                key_dim=key_dim,
                kernel_regularizer=kernel_regularizer,
            )(tensor, tensor)
            tensor = layers.add([tensor, attention])
            tensor = layers.LayerNormalization(epsilon=1e-6)(tensor)
            tensor = AveragePooling1D(pool_size=2)(tensor)

        if dropout:
            tensor = Dropout(dropout)(tensor)

        if batch_normalization:
            tensor = BatchNormalization()(tensor)

    # Add a global MaxPooling layer to reduce the dimensionality of the output if needed
    if network_type == "ATTN":
        tensor = GlobalMaxPooling1D()(tensor)

    # Add dense layers
    for units, activation in zip(ndense_layers, dense_activations):
        # Add a Dense layer with the specified number of units and activation function
        # Use the kernel regularizer if specified
        tensor = Dense(
            units,
            activation=activation,
            kernel_regularizer=kernel_regularizer,
        )(tensor)
        if dropout:
            tensor = Dropout(dropout)(tensor)
        if batch_normalization:
            tensor = BatchNormalization()(tensor)

    # Create the output layer
    outputs = tensor = Dense(num_classes, activation="softmax")(tensor)
    # Create the model
    model = Model(inputs=input_tensor, outputs=outputs)

    return model
