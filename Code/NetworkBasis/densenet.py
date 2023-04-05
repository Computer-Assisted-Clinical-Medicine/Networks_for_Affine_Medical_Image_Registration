import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dropout, Activation, Convolution2D, Convolution3D
from tensorflow.keras.layers import BatchNormalization

"""
    adapted from https://github.com/okason97/DenseNet-Tensorflow2
"""

def conv_block(x, stage, branch, nb_filter, dropout_rate=None, dim=3):
    conv_name_base = 'conv' + str(stage) + '_' + str(branch)
    relu_name_base = 'relu' + str(stage) + '_' + str(branch)

    x = BatchNormalization(name=conv_name_base + '_x2_bn')(x)
    x = Activation('relu', name=relu_name_base + '_x2')(x)
    if dim == 2:
        x = Convolution2D(nb_filter, 3, 1, name=conv_name_base + '_x2', use_bias=False, padding="SAME")(x)
    elif dim == 3:
        x = Convolution3D(nb_filter, 3, 1, name=conv_name_base + '_x2', use_bias=False, padding="SAME")(x)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)
    return x

def dense_block(x, stage, nb_layers, nb_filter, growth_rate, dropout_rate=None,
                grow_nb_filters=True, dim=3):
    concat_feat = x
    for i in range(nb_layers):
        branch = i + 1
        x = conv_block(concat_feat, stage, branch, growth_rate, dropout_rate, dim=dim)
        concat_feat = tf.concat([concat_feat, x], -1)

        if grow_nb_filters:
            nb_filter += growth_rate

    return concat_feat, nb_filter

def DenseBlocks(growth_rate=8, nb_filter=8, nb_layers = 1,dropout_rate=0.0,shape=(256,256,32,3), name='Dense', dim=3, stage=0):
    img_input = Input(shape=shape, name='data')
    x, nb_filter = dense_block(img_input, stage, nb_layers, nb_filter, growth_rate, dropout_rate=dropout_rate, dim=dim)

    return Model(inputs=img_input, outputs=x, name=name)