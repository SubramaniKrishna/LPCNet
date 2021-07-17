import numpy as np
import sys
sys.path.append('../src/')
sys.path.append('../training_tf2/')
from ulaw import *
import matplotlib.pyplot as pyp
from IPython.display import Audio,display
from scipy.signal import freqz

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GRU, Dense, Embedding, Reshape, Concatenate, Lambda, Conv1D, Multiply, Add, Bidirectional, MaxPooling1D, Activation
from tensorflow.keras.layers import Lambda, Concatenate, Layer
from tensorflow.keras import backend as K
import tensorflow as tf
import difflpc

scale = 255.0/32768.0
scale_1 = 32768.0/255.0
def tf_l2u(x):
    s = K.sign(x)
    x = K.abs(x)
    u = (s*(128*K.log(1+scale*x)/K.log(256.0)))
    u = K.clip(128 + u, 0, 255)
    return u

def tf_u2l(u):
    u = u - 128
    s = K.sign(u)
    u = K.abs(u)
    return s*scale_1*(K.exp(u/128.*K.log(256.0))-1)

scale = 255.0/32768.0
scale_1 = 32768.0/255.0
def tf_l2u(x):
    s = K.sign(x)
    x = K.abs(x)
    u = (s*(128*K.log(1+scale*x)/K.log(256.0)))
    u = K.clip(128 + u, 0, 255)
    return u

def tf_u2l(u):
    u = u - 128
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

error_calc = Lambda(lambda x: tf_l2u(tf_u2l(x[0]) - tf.roll(tf_u2l(x[1]),1,axis = 1)))

feature_file = sys.argv[1]
pcm_file = sys.argv[2]

features = np.fromfile(feature_file, dtype='float32')
data = np.fromfile(pcm_file, dtype='uint8').astype('float32')

frame_size = 160
nb_features = 55
nb_used_features = 38
feature_chunk_size = 15
pcm_chunk_size = frame_size*feature_chunk_size
nb_frames = len(data)//(4*pcm_chunk_size)

# Feature Reshaping
features = features[:nb_frames*feature_chunk_size*nb_features]
# features = np.reshape(features, (nb_frames*feature_chunk_size, nb_features))
features = np.reshape(features, (nb_frames, feature_chunk_size, nb_features))
# Data Reshaping
data = data[:nb_frames*4*pcm_chunk_size]
sig = np.reshape(data[0::4], (nb_frames, pcm_chunk_size, 1))
pred = np.reshape(data[1::4], (nb_frames, pcm_chunk_size, 1))
in_exc = np.reshape(data[2::4], (nb_frames, pcm_chunk_size, 1))
out_exc = np.reshape(data[3::4], (nb_frames, pcm_chunk_size, 1))

lpcoeffs = features[:, :, nb_used_features+1:]
features = features[:, :, :nb_used_features]
features[:,:,18:36] = 0

list_weightfiles = ['/home/ubuntu/git/LPCNet/model_weights/difflpc/rc_lar_ntt_50.h5']

for file in list_weightfiles:
    dset = file.split('/')[-1].split('_')[2]
    print(dset)
    np.random.seed(333)

    lpc_coeffs = difflpc.difflpc(training=False)
    lpc_coeffs.load_weights(file)

    # layer_name = 'rc2lpc'
    # if dset == 'mcgill':
        # layer_name = 'diff_rc2lpc_1'
    # lpc_coeffs = Model(inputs=model.input,outputs=model.get_layer(layer_name).output)

    inp_test = K.constant(features[:16,:,:])
    inp_mock = K.constant(sig[:16,:,:])
    input_sig = K.constant(out_exc[:16,:,:])
    inp_lpc = lpcoeffs[:16,:,:]
    lpc_model = lpc_coeffs(inp_test)
    model_preds = diff_pred()([inp_mock,lpc_model])
    lpc_model = lpc_model.numpy()
    model_preds = model_preds.numpy()
    res_delay = error_calc([inp_mock,model_preds])

    nauds = 5
    pyp.figure()
    for i in range(nauds):
        aud_ind = np.random.randint(0,15)
        frame_input = np.random.randint(0,15)
        x_framed = ulaw2lin(input_sig[aud_ind,frame_input*frame_size:(frame_input + 1)*frame_size,:])
        G_gt = (np.linalg.norm(x_framed - ulaw2lin(pred[aud_ind,frame_input*frame_size:(frame_input + 1)*frame_size,:])))
        lpc_frame = lpcoeffs[aud_ind,frame_input,:]
        lpcw,lpch = freqz(G_gt,np.insert(lpc_frame,0,1),frame_size//2)
        G_model = (np.linalg.norm(x_framed - ulaw2lin(model_preds[aud_ind,frame_input*frame_size:(frame_input + 1)*frame_size,:])))
        lpc_frame_model = lpc_model[aud_ind,frame_input]
        lpcw_model,lpch_model = freqz(G_model,np.insert(lpc_frame_model,0,1),frame_size//2)
        res_frame = res_delay[aud_ind,frame_input*frame_size:(frame_input + 1)*frame_size,:]
        
        pyp.subplot(nauds,4,4*i + 1)
        pyp.title("Spectral Envelopes")
        pyp.plot(np.log(np.abs(np.fft.fft(x_framed[:,0]))[:frame_size//2]),'b')
        pyp.plot(np.log(np.abs(lpch)),'g')
        pyp.plot(np.log(np.abs(lpch_model)),'r')

        pyp.subplot(nauds,4,4*i + 2)
        pyp.title(str(G_gt))
        pyp.plot(x_framed,'b')
        pyp.plot(ulaw2lin(pred[aud_ind,frame_input*frame_size:(frame_input + 1)*frame_size,:]),'g')
        pyp.plot(x_framed - ulaw2lin(pred[aud_ind,frame_input*frame_size:(frame_input + 1)*frame_size,:]),'k')

        pyp.subplot(nauds,4,4*i + 3)
        pyp.title(str(G_model))
        pyp.plot(x_framed,'b')
        pyp.plot(ulaw2lin(model_preds[aud_ind,frame_input*frame_size:(frame_input + 1)*frame_size,:]),'g')
        pyp.plot(x_framed - ulaw2lin(model_preds[aud_ind,frame_input*frame_size:(frame_input + 1)*frame_size,:]),'k')

        pyp.subplot(nauds,4,4*i + 4)
        pyp.plot((x_framed - ulaw2lin(pred[aud_ind,frame_input*frame_size:(frame_input + 1)*frame_size,:]))[:20],'k')
        pyp.plot(tf_u2l(res_frame[1:])[:20],'r')

    pyp.tight_layout()
    pyp.savefig('./LPC_plots_testing_' + dset + '_.png')