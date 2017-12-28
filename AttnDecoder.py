#! /usr/bin/env python3
# -*- coding: utf-8 -*-

from keras import backend as K
from keras.engine.topology import Layer
import pdb


class AttnDecoderCell(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        self.state_size = output_dim
        super(AttnDecoderCell, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_dim = input_shape[0][-1]
        self.input_len = input_shape[1][1]

        self.w_att = self.add_weight(
            name='w_att',
            shape=(self.output_dim * 2, 1),
            initializer='truncated_normal',
            trainable=True)
        self.b_att = self.add_weight(
            name='b_att', shape=(1, ), initializer='ones', trainable=True)

        self.w_z = self.add_weight(
            name='w_z',
            shape=(self.output_dim, self.output_dim),
            initializer='truncated_normal',
            trainable=True)
        self.u_z = self.add_weight(
            name='u_z',
            shape=(self.input_dim + self.output_dim, self.output_dim),
            initializer='truncated_normal',
            trainable=True)
        self.b_z = self.add_weight(
            name='b_z',
            shape=(self.output_dim, ),
            initializer='ones',
            trainable=True)

        self.w_r = self.add_weight(
            name='w_r',
            shape=(self.output_dim, self.output_dim),
            initializer='truncated_normal',
            trainable=True)
        self.u_r = self.add_weight(
            name='u_r',
            shape=(self.input_dim + self.output_dim, self.output_dim),
            initializer='truncated_normal',
            trainable=True)
        self.b_r = self.add_weight(
            name='b_r',
            shape=(self.output_dim, ),
            initializer='ones',
            trainable=True)

        self.w_h = self.add_weight(
            name='w_h',
            shape=(self.output_dim, self.output_dim),
            initializer='truncated_normal',
            trainable=True)
        self.u_h = self.add_weight(
            name='u_h',
            shape=(self.input_dim + self.output_dim, self.output_dim),
            initializer='truncated_normal',
            trainable=True)
        self.b_h = self.add_weight(
            name='b_h',
            shape=(self.output_dim, ),
            initializer='ones',
            trainable=True)

        super(AttnDecoderCell, self).build(input_shape)

    def call(self, inputs, state, constants=None, **kwargs):
        state = state[0]
        constants = constants[0]

        st = K.repeat(state, self.input_len)
        energy = K.concatenate([st, constants])
        energy = K.bias_add(K.dot(energy, self.w_att), self.b_att)
        energy = K.permute_dimensions(energy, (2, 0, 1))
        energy = energy[0]
        alpha = K.softmax(energy)
        alpha = K.repeat(alpha, self.output_dim)
        alpha = K.permute_dimensions(alpha, (0, 2, 1))
        v = K.sum(constants * alpha, axis=1)
        inputs = K.concatenate([inputs, v])

        z_t = K.dot(state, self.w_z) + K.dot(inputs, self.u_z)
        z_t = K.sigmoid(K.bias_add(z_t, self.b_z))

        r_t = K.dot(state, self.w_r) + K.dot(inputs, self.u_r)
        r_t = K.sigmoid(K.bias_add(r_t, self.b_r))

        h = K.dot((state * r_t), self.w_h) + K.dot(inputs, self.u_h)
        h = K.tanh(K.bias_add(h, self.b_h))

        s_t = (1 - z_t) * h + z_t * state
        s_t = s_t + v

        return s_t, [s_t]

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)


if __name__ == '__main__':
    import numpy as np
    from keras.layers import Input, RNN
    from keras.models import Model
    tar_seq = np.ones((100, 30, 300))
    enc_seq = np.ones((100, 30, 300))
    _h_ = np.ones((100, 300))

    cell = AttnDecoderCell(300)
    attn_layer = RNN(cell, return_sequences=True)

    inp_enc = Input(shape=(30, 300))
    inp_tar = Input(shape=(30, 300))
    inp_h = Input(shape=(300, ))

    out = attn_layer(inp_tar, initial_state=inp_h, constants=[inp_enc])
    model = Model([inp_enc, inp_tar, inp_h], out)
    pre = model.predict([enc_seq, tar_seq, _h_])
    print(pre)
