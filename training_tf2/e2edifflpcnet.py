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
import difflpc

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
    pcm = Input(shape=(None, 3))
    feat = Input(shape=(None, nb_used_features))
    pitch = Input(shape=(None, 1))
    dec_feat = Input(shape=(None, 128))
    dec_state1 = Input(shape=(rnn_units1,))
    dec_state2 = Input(shape=(rnn_units2,))
    Input_extractor = Lambda(lambda x: K.expand_dims(x[0][:,:,x[1]],axis = -1))
    error_calc = Lambda(lambda x: tf.roll(x[0],1,axis = -1) - x[1])
    padding = 'valid' if training else 'same'
    fconv1 = Conv1D(128, 3, padding=padding, activation='tanh', name='feature_conv1')
    fconv2 = Conv1D(128, 3, padding=padding, activation='tanh', name='feature_conv2')

    # Extracting LPCoeffs from Features and obtaining prediction
    feat2lpc = difflpc.difflpc(training = True)
    lpcoeffs = feat2lpc(feat)
    tensor_preds = diff_pred(name = "lpc2preds")([Input_extractor([pcm,0]),lpcoeffs])
    past_errors = error_calc([Input_extractor([pcm,0]),tensor_preds])

    embed = diff_Embed(name='embed_sig')
    # print(pcm.shape,embed(pcm).shape)
    # tensor_preds = diff_pred(name = 'diffpred')([Input_extractor([pcm,0]),lpcoeffs])
    cpcm = Concatenate()([Input_extractor([pcm,0]),tensor_preds,past_errors])
    cpcm = Reshape((-1, embed_size*3))(embed(cpcm))
    cpcm_decoder = Concatenate()([Input_extractor([pcm,0]),Input_extractor([pcm,1]),Input_extractor([pcm,2])])
    cpcm_decoder = Reshape((-1, embed_size*3))(embed(cpcm_decoder))
    # cpcm = Concatenate()([iT(Input_extractor([pcm,0])),iT(tensor_preds),iT(Input_extractor([pcm,2]))])
    # cpcm_decoder = Concatenate()([iT(Input_extractor([pcm,0])),iT(Input_extractor([pcm,1])),iT(Input_extractor([pcm,2]))])

    pembed = Embedding(256, 64, name='embed_pitch')
    cat_feat = Concatenate()([feat, Reshape((-1, 64))(pembed(pitch))])
    
    cfeat = fconv2(fconv1(cat_feat))

    fdense1 = Dense(128, activation='tanh', name='feature_dense1')
    fdense2 = Dense(128, activation='tanh', name='feature_dense2')

    cfeat = fdense2(fdense1(cfeat))
    
    rep = Lambda(lambda x: K.repeat_elements(x, frame_size, 1))

    quant = quant_regularizer if quantize else None

    if training:
        rnn = CuDNNGRU(rnn_units1, return_sequences=True, return_state=True, name='gru_a',
              recurrent_constraint = constraint, recurrent_regularizer=quant)
        rnn2 = CuDNNGRU(rnn_units2, return_sequences=True, return_state=True, name='gru_b',
               kernel_constraint=constraint, kernel_regularizer=quant)
    else:
        rnn = GRU(rnn_units1, return_sequences=True, return_state=True, recurrent_activation="sigmoid", reset_after='true', name='gru_a',
              recurrent_constraint = constraint, recurrent_regularizer=quant)
        rnn2 = GRU(rnn_units2, return_sequences=True, return_state=True, recurrent_activation="sigmoid", reset_after='true', name='gru_b',
               kernel_constraint=constraint, kernel_regularizer=quant)

    rnn_in = Concatenate()([cpcm, rep(cfeat)])
    md = MDense(pcm_levels, activation='softmax', name='dual_fc')
    gru_out1, _ = rnn(rnn_in)
    gru_out2, _ = rnn2(Concatenate()([gru_out1, rep(cfeat)]))
    ulaw_prob = md(gru_out2)
    
    if adaptation:
        rnn.trainable=False
        rnn2.trainable=False
        md.trainable=False
        embed.Trainable=False
    
    m_out = Concatenate()([tensor_preds,ulaw_prob])
    model = Model([pcm, feat, pitch], m_out)
    model.rnn_units1 = rnn_units1
    model.rnn_units2 = rnn_units2
    model.nb_used_features = nb_used_features
    model.frame_size = frame_size

    encoder = Model([feat, pitch], [cfeat,lpcoeffs])
    
    dec_rnn_in = Concatenate()([cpcm_decoder, dec_feat])
    dec_gru_out1, state1 = rnn(dec_rnn_in, initial_state=dec_state1)
    dec_gru_out2, state2 = rnn2(Concatenate()([dec_gru_out1, dec_feat]), initial_state=dec_state2)
    dec_ulaw_prob = md(dec_gru_out2)

    decoder = Model([pcm, dec_feat, dec_state1, dec_state2], [dec_ulaw_prob, state1, state2])
    return model, encoder, decoder

class diff_Embed(Layer):

    def __init__(self, units=128, dict_size = 256, **kwargs):
        super(diff_Embed, self).__init__(**kwargs)
        self.units = units
        self.dict_size = dict_size

    def build(self, input_shape):  # Create the state of the layer (weights)
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(initial_value=w_init(shape=(self.dict_size, self.units),dtype='float32'),trainable=True)

    def call(self, inputs):  # Defines the computation from inputs to outputs
        alpha = inputs - tf.math.floor(inputs)
        alpha = tf.expand_dims(alpha,axis = -1)
        alpha = tf.tile(alpha,[1,1,1,self.units])
        inputs = tf.cast(inputs,'int32')
        ip_E = tf.one_hot(inputs, self.dict_size)
        # ip_Ep1 = tf.one_hot(tf.clip_by_value(inputs + 1, 0, 255), self.dict_size)
        # M = (1 - alpha)*tf.matmul(ip_E, self.w) + alpha*(tf.matmul(ip_Ep1, self.w))
        M = (1 - alpha)*tf.gather(self.w,inputs) + alpha*tf.gather(self.w,tf.clip_by_value(inputs + 1, 0, 255))
        #   print(ip_E[0,:,0,0])
        #   print(ip_E.shape,self.w.shape)
        # print(tf.matmul(ip_E, self.w).shape)
        return M

    def get_config(self):
        config = super(diff_Embed, self).get_config()
        config.update({"units": self.units})
        config.update({"dict_size" : self.dict_size})
        return config

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

lpcoeffs_N = 16
class diff_pred(Layer):
    def call(self, inputs):
        xt = tf_u2l(inputs[0])
        lpc = inputs[1]

        rept = Lambda(lambda x: K.repeat_elements(x , frame_size, 1))
        zpX = Lambda(lambda x: K.concatenate([0*x[:,0:lpcoeffs_N,:], x],axis = 1))
        cX = Lambda(lambda x: K.concatenate([x[:,(lpcoeffs_N - i):(lpcoeffs_N - i + 2400),:] for i in range(lpcoeffs_N)],axis = 2))
        
        pred = -Multiply()([rept(lpc),cX(zpX(xt))])

        return tf_l2u(K.sum(pred,axis = 2,keepdims = True))
