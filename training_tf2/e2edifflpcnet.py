#!/usr/bin/python3
'''Copyright (c) 2018 Mozilla

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions
   are met:

   - Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

   - Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE FOUNDATION OR
   CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''

import math
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GRU, Dense, Embedding, Reshape, Concatenate, Lambda, Conv1D, Multiply, Add, Bidirectional, MaxPooling1D, Activation, Layer
from tensorflow.compat.v1.keras.layers import CuDNNGRU
from tensorflow.keras import backend as K
from tensorflow.keras.constraints import Constraint
from tensorflow.keras.initializers import Initializer
from tensorflow.keras.callbacks import Callback
from mdense import MDense
import numpy as np
import h5py
import sys

frame_size = 160
pcm_bits = 8
embed_size = 128
pcm_levels = 2**pcm_bits

def quant_regularizer(x):
    Q = 128
    Q_1 = 1./Q
    #return .01 * tf.reduce_mean(1 - tf.math.cos(2*3.1415926535897931*(Q*x-tf.round(Q*x))))
    return .01 * tf.reduce_mean(K.sqrt(K.sqrt(1.0001 - tf.math.cos(2*3.1415926535897931*(Q*x-tf.round(Q*x))))))

class Sparsify(Callback):
    def __init__(self, t_start, t_end, interval, density):
        super(Sparsify, self).__init__()
        self.batch = 0
        self.t_start = t_start
        self.t_end = t_end
        self.interval = interval
        self.final_density = density

    def on_batch_end(self, batch, logs=None):
        #print("batch number", self.batch)
        self.batch += 1
        if self.batch < self.t_start or ((self.batch-self.t_start) % self.interval != 0 and self.batch < self.t_end):
            #print("don't constrain");
            pass
        else:
            #print("constrain");
            layer = self.model.get_layer('gru_a')
            w = layer.get_weights()
            p = w[1]
            nb = p.shape[1]//p.shape[0]
            N = p.shape[0]
            #print("nb = ", nb, ", N = ", N);
            #print(p.shape)
            #print ("density = ", density)
            for k in range(nb):
                density = self.final_density[k]
                if self.batch < self.t_end:
                    r = 1 - (self.batch-self.t_start)/(self.t_end - self.t_start)
                    density = 1 - (1-self.final_density[k])*(1 - r*r*r)
                A = p[:, k*N:(k+1)*N]
                A = A - np.diag(np.diag(A))
                #This is needed because of the CuDNNGRU strange weight ordering
                A = np.transpose(A, (1, 0))
                L=np.reshape(A, (N//4, 4, N//8, 8))
                S=np.sum(L*L, axis=-1)
                S=np.sum(S, axis=1)
                SS=np.sort(np.reshape(S, (-1,)))
                thresh = SS[round(N*N//32*(1-density))]
                mask = (S>=thresh).astype('float32');
                mask = np.repeat(mask, 4, axis=0)
                mask = np.repeat(mask, 8, axis=1)
                mask = np.minimum(1, mask + np.diag(np.ones((N,))))
                #This is needed because of the CuDNNGRU strange weight ordering
                mask = np.transpose(mask, (1, 0))
                p[:, k*N:(k+1)*N] = p[:, k*N:(k+1)*N]*mask
                #print(thresh, np.mean(mask))
            w[1] = p
            layer.set_weights(w)
            

class PCMInit(Initializer):
    def __init__(self, gain=.1, seed=None):
        self.gain = gain
        self.seed = seed

    def __call__(self, shape, dtype=None):
        num_rows = 1
        for dim in shape[:-1]:
            num_rows *= dim
        num_cols = shape[-1]
        flat_shape = (num_rows, num_cols)
        if self.seed is not None:
            np.random.seed(self.seed)
        a = np.random.uniform(-1.7321, 1.7321, flat_shape)
        #a[:,0] = math.sqrt(12)*np.arange(-.5*num_rows+.5,.5*num_rows-.4)/num_rows
        #a[:,1] = .5*a[:,0]*a[:,0]*a[:,0]
        a = a + np.reshape(math.sqrt(12)*np.arange(-.5*num_rows+.5,.5*num_rows-.4)/num_rows, (num_rows, 1))
        return self.gain * a

    def get_config(self):
        return {
            'gain': self.gain,
            'seed': self.seed
        }

class WeightClip(Constraint):
    '''Clips the weights incident to each hidden unit to be inside a range
    '''
    def __init__(self, c=2):
        self.c = c

    def __call__(self, p):
        # Ensure that abs of adjacent weights don't sum to more than 127. Otherwise there's a risk of
        # saturation when implementing dot products with SSSE3 or AVX2.
        return self.c*p/tf.maximum(self.c, tf.repeat(tf.abs(p[:, 1::2])+tf.abs(p[:, 0::2]), 2, axis=1))
        #return K.clip(p, -self.c, self.c)

    def get_config(self):
        return {'name': self.__class__.__name__,
            'c': self.c}

constraint = WeightClip(0.992)

def new_lpcnet_model(rnn_units1=384, rnn_units2=16, nb_used_features = 38, training=False, adaptation=False, quantize=False):
