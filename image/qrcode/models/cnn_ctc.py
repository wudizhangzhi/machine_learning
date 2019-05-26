import os
import itertools
import datetime
import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Input, Dense, Activation
from keras.layers import Reshape, Lambda
from keras.layers.merge import add, concatenate
from keras.models import Model
from keras.layers.recurrent import GRU
from keras.optimizers import SGD


def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


def get_model(img_w, img_h, channel, output_size, absolute_max_string_len,
              conv_filters=16, kernel_size=(3, 3), pool_size=2, time_dense_size=32, rnn_size=512,
              is_train=False):
    if K.image_data_format() == 'channels_first':
        input_shape = (channel, img_w, img_h)
    else:
        #     input_shape = (img_w, img_h, channel)
        input_shape = (img_h, img_w, channel)

    act = 'relu'
    input_data = Input(name='the_input', shape=input_shape, dtype='float32')
    inner = Conv2D(conv_filters, kernel_size, padding='same',
                   activation=act, kernel_initializer='he_normal',
                   name='conv1')(input_data)
    inner = MaxPooling2D(pool_size=(pool_size, pool_size), name='max1')(inner)
    inner = Conv2D(conv_filters, kernel_size, padding='same',
                   activation=act, kernel_initializer='he_normal',
                   name='conv2')(inner)
    inner = MaxPooling2D(pool_size=(pool_size, pool_size), name='max2')(inner)

    conv_to_rnn_dims = (img_w // (pool_size ** 2), (img_h // (pool_size ** 2)) * conv_filters)
    inner = Reshape(target_shape=conv_to_rnn_dims, name='reshape')(inner)

    # cuts down input size going into RNN:
    inner = Dense(time_dense_size, activation=act, name='dense1')(inner)

    # Two layers of bidirectional GRUs
    # GRU seems to work as well, if not better than LSTM:
    gru_1 = GRU(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru1')(inner)
    gru_1b = GRU(rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru1_b')(
        inner)
    gru1_merged = add([gru_1, gru_1b])
    gru_2 = GRU(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru2')(gru1_merged)
    gru_2b = GRU(rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru2_b')(
        gru1_merged)

    # transforms RNN output to character activations:
    inner = Dense(output_size, kernel_initializer='he_normal',
                  name='dense2')(concatenate([gru_2, gru_2b]))
    y_pred = Activation('softmax', name='softmax')(inner)
    base_model = Model(input=input_data, output=y_pred)
    # base_model.summary()

    labels = Input(name='the_labels', shape=[absolute_max_string_len], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')
    # Keras doesn't currently support loss funcs with extra parameters
    # so CTC loss is implemented in a lambda layer
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])

    model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)
    test_func = K.function([input_data], [y_pred])
    return model, test_func


def decode_batch(test_func, labels_to_text_func, word_batch):
    """
    将矩阵转化为字符串
    :param test_func:
    :param labels_to_text_func:
    :param word_batch:
    :return:
    """
    out = test_func([word_batch])[0]
    ret = []
    for j in range(out.shape[0]):
        out_best = list(np.argmax(out[j, 2:], 1))
        out_best = [k for k, g in itertools.groupby(out_best)]
        outstr = labels_to_text_func(out_best)
        ret.append(outstr)
    return ret


LABELS = '0123456789abcdefghijklmnopqrstuvwxyz '
config = {
    'img_w': 69,
    'img_h': 24,
    'channel': 1,
    'output_size': len(LABELS) + 1,
    'absolute_max_string_len': 4,
    'conv_filters': 16,
    'kernel_size': (3, 3),
    'pool_size': 2,
    'time_dense_size': 32,
    'rnn_size': 512
}
icp_config = {
    'img_w': 200,
    'img_h': 60,
    'channel': 1,
    'output_size': len(LABELS) + 1,
    'absolute_max_string_len': 6,
    'conv_filters': 16,
    'kernel_size': (3, 3),
    'pool_size': 2,
    'time_dense_size': 32,
    'rnn_size': 512
}


def junka_model():
    return get_model(**config)


def icp_model():
    return get_model(**icp_config)


def train(model, train_gen, validate_gen,
          steps_per_epoch=100, epochs=40, validation_steps=5,
          optimizer=None,
          callbacks=None):
    # clipnorm seems to speeds up convergence
    if not optimizer:
        optimizer = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=optimizer)
    model.fit_generator(generator=train_gen,
                        steps_per_epoch=steps_per_epoch,
                        epochs=epochs,
                        validation_data=validate_gen,
                        validation_steps=validation_steps,
                        callbacks=callbacks,
                        verbose=1)


if __name__ == '__main__':
    junka_model()
