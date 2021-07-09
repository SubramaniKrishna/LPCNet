import difflpc
import sys
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Input, GRU, Dense, Embedding, Reshape, Concatenate, Lambda, Conv1D, Multiply, Add, Bidirectional, MaxPooling1D, Activation, Layer, LeakyReLU
from ulaw import *
import tensorflow.keras.backend as K
import h5py

import tensorflow as tf
lpcoeffs_N = 16
# Custom Loss Function
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

def mse_rc():
    def loss(y_true,y_pred):
        # y_true = diff_lpc2rc()(y_true)
        squared_difference = K.square(y_true - y_pred)
        return tf.reduce_mean(squared_difference, axis=-1)
    return loss

def rc_pred_mse(alpha = 0.5):
    def loss(y_true,y_pred):
        L2_rc = diff_lpc2rc()(y_true[1])
        L2_rc = K.square(L2_rc - y_pred[1])

        L2_pred = K.square(y_true[0] - y_pred[0])

        L2_sum = alpha*L2_rc + (1 - alpha)*L2_pred
        return tf.reduce_mean(L2_sum, axis=-1)
    return loss

def normalized_mse():
    def loss(y_true,y_pred):
        L2_pred = K.square(y_true - y_pred)
        # norms = K.square(y_true) + 1.0e-8
        # L2_normalized = L2_pred
        return tf.reduce_mean(L2_pred, axis=-1)
    return loss
# from tensorflow.keras.backend.tensorflow_backend import set_session
# config = tf.ConfigProto()

# use this option to reserve GPU memory, e.g. for running more than
# one thing at a time.  Best to disable for GPUs with small memory
# config.gpu_options.per_process_gpu_memory_fraction = 0.44

# set_session(tf.Session(config=config))

nb_epochs = 10

# Try reducing batch_size if you run out of memory on your GPU
batch_size = 64

model = difflpc.difflpc(training=True)
# model = diff_lpc.cc2rc()

# model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])
# model.summary()

# feature_file = sys.argv[1]
# pcm_file = sys.argv[2]     # 16 bit unsigned short PCM samples
feature_file = './ntt_modloss_features.s16'
pcm_file = './ntt_modloss_data.u8'
dset = 'ntt'

frame_size = model.frame_size
nb_features = 55
nb_used_features = model.nb_used_features
feature_chunk_size = 15
pcm_chunk_size = frame_size*feature_chunk_size

# u for unquantised, load 16 bit PCM samples and convert to mu-law

data = np.fromfile(pcm_file, dtype='uint8').astype("float32")
nb_frames = len(data)//(4*pcm_chunk_size)

features = np.fromfile(feature_file, dtype='float32')

# limit to discrete number of frames
data = data[:nb_frames*4*pcm_chunk_size]
features = features[:nb_frames*feature_chunk_size*nb_features]

features = np.reshape(features, (nb_frames*feature_chunk_size, nb_features))

sig = np.reshape(data[0::4], (nb_frames, pcm_chunk_size, 1))
pred = np.reshape(data[1::4], (nb_frames, pcm_chunk_size, 1))
in_exc = np.reshape(data[2::4], (nb_frames, pcm_chunk_size, 1))
out_exc = np.reshape(data[3::4], (nb_frames, pcm_chunk_size, 1))

features = np.reshape(features, (nb_frames, feature_chunk_size, nb_features))
lpcoeffs = features[:, :, nb_used_features+1:]
features = features[:, :, :nb_used_features]
features[:,:,18:36] = 0

# fpad1 = np.concatenate([features[0:1, 0:2, :], features[:-1, -2:, :]], axis=0)
# fpad2 = np.concatenate([features[1:, :2, :], features[0:1, -2:, :]], axis=0)
# features = np.concatenate([fpad1, features, fpad2], axis=1)
# lpad1 = np.concatenate([lpcoeffs[0:1, 0:2, :], lpcoeffs[:-1, -2:, :]], axis=0)
# lpad2 = np.concatenate([lpcoeffs[1:, :2, :], lpcoeffs[0:1, -2:, :]], axis=0)
# lpcoeffs = np.concatenate([lpad1, lpcoeffs, lpad2], axis=1)

periods = (.1 + 50*features[:,:,36:37]+100).astype('int16')

# print(sig.shape,pred.shape,in_exc.shape,periods.shape)
# in_data = np.concatenate([sig, pred, in_exc], axis=-1)
# print("IDS",in_data.shape)
# del sig
# del pred
# del in_exc

# dump models to disk as we go
dir_w = './model_weights/difflpc/'
checkpoint = ModelCheckpoint(dir_w + 'rcplusmse_test_' + dset + '_{epoch:02d}.h5')

# #Set this to True to adapt an existing model (e.g. on new data)
# adaptation = False

# if adaptation:
#     #Adapting from an existing model
#     model.load_weights('lpcnet24c_384_10_G16_120.h5')
#     sparsify = lpcnet.Sparsify(0, 0, 1, (0.05, 0.05, 0.2))
#     lr = 0.0001
#     decay = 0
# else:
#     #Training from scratch
#     sparsify = lpcnet.Sparsify(2000, 40000, 400, (0.05, 0.05, 0.2))
#     lr = 0.001
#     decay = 5e-5

losses = {
	"rc2lpc": mse_rc(),
	"prediction": normalized_mse(),
}
lossWeights = {"rc2lpc": 0.0, "prediction": 1.0}

lr = 0.01
decay = 5e-5
model.compile(optimizer=Adam(lr, amsgrad=True, decay=decay), loss=losses,loss_weights = lossWeights)
model.save_weights(dir_w + 'rcplusmse_test_' + dset + '_00.h5')
model.fit([sig, features], [lpcoeffs,out_exc], batch_size=batch_size, epochs=nb_epochs, validation_split=0.0, callbacks=[checkpoint])
# print(in_data)
# OG
# model.fit([in_data, features, periods], out_exc, batch_size=batch_size, epochs=nb_epochs, validation_split=0.0, callbacks=[checkpoint, sparsify])
# Modified
# Normalize sig,pred,in_exc
# sig = (sig - 128)*(2.0/255)
# pred = (pred - 128)*(2.0/255)
# in_exc = (in_exc - 128)*(2.0/255)
# out_exc = (out_exc - 128)*(2.0/255)
# print(features.shape,lpcoeffs.shape)
# model.fit([sig, pred, in_exc, features, periods], out_exc, batch_size=batch_size, epochs=nb_epochs, validation_split=0.0, callbacks=[checkpoint, sparsify])


# Predictions inside differentiable loop
"""
Inputs: features only
"""
# print(sig.shape,lpcoeffs.shape)
# model.fit([sig, features], [lpcoeffs,np.concatenate((sig,np.zeros((sig.shape[0],1,1))),axis = 1)[:,1:,:]], batch_size=batch_size, epochs=nb_epochs, validation_split=0.0, callbacks=[checkpoint])

# CC2RC
# model.fit([sig, features], [np.concatenate((sig,np.zeros((sig.shape[0],1,1))),axis = 1)[:,1:,:]], batch_size=batch_size, epochs=nb_epochs, validation_split=0.0, callbacks=[checkpoint])
