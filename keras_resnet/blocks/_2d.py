# -*- coding: utf-8 -*-

"""
keras_resnet.blocks._2d
~~~~~~~~~~~~~~~~~~~~~~~

This module implements a number of popular two-dimensional residual blocks.
"""

# import keras.layers
# import keras.regularizers

import keras_resnet.layers
from tensorflow.keras import backend
from tensorflow.keras.layers import Activation, Add, Conv2D, MaxPooling2D, ZeroPadding2D


parameters = {"kernel_initializer": "he_normal"}


def basic_2d(
    filters,
    stage=0,
    block=0,
    kernel_size=3,
    numerical_name=False,
    stride=None,
    freeze_bn=False,
):
    """
    A two-dimensional basic block.

    :param filters: the output’s feature space

    :param stage: int representing the stage of this block (starting from 0)

    :param block: int representing this block (starting from 0)

    :param kernel_size: size of the kernel

    :param numerical_name: if true, uses numbers to represent blocks instead of chars (ResNet{101, 152, 200})

    :param stride: int representing the stride used in the shortcut and the first conv layer, default derives stride from block id

    :param freeze_bn: if true, freezes BatchNormalization layers (ie. no updates are done in these layers)

    Usage:

        >>> import keras_resnet.blocks

        >>> keras_resnet.blocks.basic_2d(64)
    """
    if stride is None:
        if block != 0 or stage == 0:
            stride = 1
        else:
            stride = 2

    if backend.image_data_format() == "channels_last":
        axis = 3
    else:
        axis = 1

    if block > 0 and numerical_name:
        block_char = "b{}".format(block)
    else:
        block_char = chr(ord("a") + block)

    stage_char = str(stage + 2)

    def f(x):
        y = ZeroPadding2D(
            padding=1, name="padding{}{}_branch2a".format(stage_char, block_char)
        )(x)

        y = Conv2D(
            filters,
            kernel_size,
            strides=stride,
            use_bias=False,
            name="res{}{}_branch2a".format(stage_char, block_char),
            **parameters
        )(y)

        y = keras_resnet.layers.BatchNormalization(
            axis=axis,
            epsilon=1e-5,
            freeze=freeze_bn,
            name="bn{}{}_branch2a".format(stage_char, block_char),
        )(y)

        y = Activation(
            "relu", name="res{}{}_branch2a_relu".format(stage_char, block_char)
        )(y)

        y = ZeroPadding2D(
            padding=1, name="padding{}{}_branch2b".format(stage_char, block_char)
        )(y)

        y = Conv2D(
            filters,
            kernel_size,
            use_bias=False,
            name="res{}{}_branch2b".format(stage_char, block_char),
            **parameters
        )(y)

        y = keras_resnet.layers.BatchNormalization(
            axis=axis,
            epsilon=1e-5,
            freeze=freeze_bn,
            name="bn{}{}_branch2b".format(stage_char, block_char),
        )(y)

        if block == 0:
            shortcut = Conv2D(
                filters,
                (1, 1),
                strides=stride,
                use_bias=False,
                name="res{}{}_branch1".format(stage_char, block_char),
                **parameters
            )(x)

            shortcut = keras_resnet.layers.BatchNormalization(
                axis=axis,
                epsilon=1e-5,
                freeze=freeze_bn,
                name="bn{}{}_branch1".format(stage_char, block_char),
            )(shortcut)
        else:
            shortcut = x

        y = Add(name="res{}{}".format(stage_char, block_char))([y, shortcut])

        y = Activation("relu", name="res{}{}_relu".format(stage_char, block_char))(y)

        return y

    return f


def bottleneck_2d(
    filters,
    stage=0,
    block=0,
    kernel_size=3,
    numerical_name=False,
    stride=None,
    freeze_bn=False,
):
    """
    A two-dimensional bottleneck block.

    :param filters: the output’s feature space

    :param stage: int representing the stage of this block (starting from 0)

    :param block: int representing this block (starting from 0)

    :param kernel_size: size of the kernel

    :param numerical_name: if true, uses numbers to represent blocks instead of chars (ResNet{101, 152, 200})

    :param stride: int representing the stride used in the shortcut and the first conv layer, default derives stride from block id

    :param freeze_bn: if true, freezes BatchNormalization layers (ie. no updates are done in these layers)

    Usage:

        >>> import keras_resnet.blocks

        >>> keras_resnet.blocks.bottleneck_2d(64)
    """
    if stride is None:
        if block != 0 or stage == 0:
            stride = 1
        else:
            stride = 2

    if backend.image_data_format() == "channels_last":
        axis = 3
    else:
        axis = 1

    if block > 0 and numerical_name:
        block_char = "b{}".format(block)
    else:
        block_char = chr(ord("a") + block)

    stage_char = str(stage + 2)

    def f(x):
        y = Conv2D(
            filters,
            (1, 1),
            strides=stride,
            use_bias=False,
            name="res{}{}_branch2a".format(stage_char, block_char),
            **parameters
        )(x)

        y = keras_resnet.layers.BatchNormalization(
            axis=axis,
            epsilon=1e-5,
            freeze=freeze_bn,
            name="bn{}{}_branch2a".format(stage_char, block_char),
        )(y)

        y = Activation(
            "relu", name="res{}{}_branch2a_relu".format(stage_char, block_char)
        )(y)

        y = ZeroPadding2D(
            padding=1, name="padding{}{}_branch2b".format(stage_char, block_char)
        )(y)

        y = Conv2D(
            filters,
            kernel_size,
            use_bias=False,
            name="res{}{}_branch2b".format(stage_char, block_char),
            **parameters
        )(y)

        y = keras_resnet.layers.BatchNormalization(
            axis=axis,
            epsilon=1e-5,
            freeze=freeze_bn,
            name="bn{}{}_branch2b".format(stage_char, block_char),
        )(y)

        y = Activation(
            "relu", name="res{}{}_branch2b_relu".format(stage_char, block_char)
        )(y)

        y = Conv2D(
            filters * 4,
            (1, 1),
            use_bias=False,
            name="res{}{}_branch2c".format(stage_char, block_char),
            **parameters
        )(y)

        y = keras_resnet.layers.BatchNormalization(
            axis=axis,
            epsilon=1e-5,
            freeze=freeze_bn,
            name="bn{}{}_branch2c".format(stage_char, block_char),
        )(y)

        if block == 0:
            shortcut = Conv2D(
                filters * 4,
                (1, 1),
                strides=stride,
                use_bias=False,
                name="res{}{}_branch1".format(stage_char, block_char),
                **parameters
            )(x)

            shortcut = keras_resnet.layers.BatchNormalization(
                axis=axis,
                epsilon=1e-5,
                freeze=freeze_bn,
                name="bn{}{}_branch1".format(stage_char, block_char),
            )(shortcut)
        else:
            shortcut = x

        y = Add(name="res{}{}".format(stage_char, block_char))([y, shortcut])

        y = Activation("relu", name="res{}{}_relu".format(stage_char, block_char))(y)

        return y

    return f
