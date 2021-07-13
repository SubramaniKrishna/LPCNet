import math
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GRU, Dense, Embedding, Reshape, Concatenate, Lambda, Conv1D, Multiply, Add, Bidirectional, MaxPooling1D, Activation, Layer, LeakyReLU
from tensorflow.compat.v1.keras.layers import CuDNNGRU
from tensorflow.keras import backend as K
from tensorflow.keras.initializers import Initializer
from tensorflow.keras.callbacks import Callback
# from mdense import MDense
import tensorflow as tf
import numpy as np
import h5py
import sys

frame_size = 160
pcm_bits = 8
embed_size = 21
pcm_levels = 2**pcm_bits
lpcoeffs_N = 16

def cc2rc(nb_used_features = 38):
    feat = Input(shape=(None, nb_used_features))
    A = LeakyReLU()
    L1 = Dense(100, activation='linear')
    L2 = Dense(75, activation='linear')
    L3 = Dense(lpcoeffs_N, activation='linear')
    rc = L3(A(L2(A(L1(feat)))))
    L4 = Dense(lpcoeffs_N, activation='tanh')
    rc = L4(rc)

    model = Model(feat, rc)
    model.nb_used_features = nb_used_features
    model.frame_size = frame_size

    return model


def difflpc(nb_used_features = 38, training=False):
    # Input is \mu law encoded and BFCC features
    pcm_1 = Input(shape=(None, 1)) #x_t
    feat = Input(shape=(None, nb_used_features)) # BFCC
    padding = 'valid' if training else 'same'
    L1 = Conv1D(100, 3, padding=padding, activation='tanh', name='f2rc_conv1')
    L2 = Conv1D(75, 3, padding=padding, activation='tanh', name='f2rc_conv2')
    # L1 = Dense(100, activation='tanh')
    # L2 = Dense(75, activation='tanh')
    L3 = Dense(50, activation='tanh',name = 'f2rc_dense3')
    L4 = Dense(lpcoeffs_N, activation='tanh',name = "f2rc_dense4_outp_rc")
    rc = L4(L3(L2(L1(feat))))
    # Differentiable RC 2 LPC
    lpcoeffs = diff_rc2lpc(name = "rc2lpc")(rc)
    # preds = diff_pred(name="prediction")([pcm_1,lpcoeffs])

    model = Model(feat,lpcoeffs,name = 'f2lpc')
    model.nb_used_features = nb_used_features
    model.frame_size = frame_size
    return model


scale = 255.0/32768.0
scale_1 = 32768.0/255.0
def tf_l2u(x):
    s = K.sign(x)
    x = K.abs(x)
    u = (s*(128*K.log(1+scale*x)/K.log(256.0)))
    u = K.clip(128 + u, 0, 255)
    return u

def tf_u2l(u):
    u = u - 128.0
    s = K.sign(u)
    u = K.abs(u)
    return s*scale_1*(K.exp(u/128.*K.log(256.0))-1)

class diff_pred(Layer):
    def call(self, inputs):
        xt = tf_u2l(inputs[0])
        lpc = inputs[1]

        rept = Lambda(lambda x: K.repeat_elements(x , frame_size, 1))
        zpX = Lambda(lambda x: K.concatenate([0*x[:,0:lpcoeffs_N,:], x],axis = 1))
        cX = Lambda(lambda x: K.concatenate([x[:,(lpcoeffs_N - i):(lpcoeffs_N - i + 2400),:] for i in range(lpcoeffs_N)],axis = 2))
        
        pred = -Multiply()([rept(lpc),cX(zpX(xt))])

        return tf_l2u(K.sum(pred,axis = 2,keepdims = True))

class diff_rc2lpc(Layer):
    def call(self, inputs):
        def pred_lpc_recursive(input):
            temp = (input[0] + K.repeat_elements(input[1],input[0].shape[2],2)*K.reverse(input[0],axes = 2))
            temp = Concatenate(axis = 2)([temp,input[1]])
            return temp
        Llpc = Lambda(pred_lpc_recursive)
        lpc_init = inputs
        for i in range(1,lpcoeffs_N):
            lpc_init = Llpc([lpc_init[:,:,:i],K.expand_dims(inputs[:,:,i],axis = -1)])
        return lpc_init

class diff_lpc2rc(Layer):
    def call(self, inputs):
        def pred_rc_recursive(input):
            ki = K.repeat_elements(K.expand_dims(input[1][:,:,0],axis = -1),input[0].shape[2],2)
            temp = (input[0] - ki*K.reverse(input[0],axes = 2))/(1 - ki*ki)
            temp = Concatenate(axis = 2)([temp,input[1]])
            return temp
        Lrc = Lambda(pred_rc_recursive)
        rc_init = inputs
        for i in range(1,lpcoeffs_N):
            j = (lpcoeffs_N - i + 1)
            rc_init = Lrc([rc_init[:,:,:(j - 1)],rc_init[:,:,(j - 1):]])
        return rc_init
            