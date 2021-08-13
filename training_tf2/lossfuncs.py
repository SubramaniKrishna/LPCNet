"""
Custom Loss functions and metrics for training/analysis
"""

from tf_funcs import *
import tensorflow as tf

# The following loss functions all expect the lpcnet model to output the lpc prediction

# Computing the excitation by subtracting the lpc prediction from the target, followed by minimizing the cross entropy
def res_from_sigloss():
    def loss(y_true,y_pred):
        p = y_pred[:,:,0:1]
        model_out = y_pred[:,:,1:]
        e_gt = tf_l2u(tf_u2l(y_true) - tf_u2l(p))
        e_gt = tf.round(e_gt)
        e_gt = tf.cast(e_gt,'int32')
        sparse_cel = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)(e_gt,model_out)
        return sparse_cel
    return loss

# Interpolated and Compensated Loss (In case of end to end lpcnet)
# Interpolates between adjacent embeddings based on the fractional value of the excitation computed (similar to the embedding interpolation)
# Also adds a probability compensation (to account for matching cross entropy in the linear domain), weighted by gamma
def interp_mulaw(gamma = 1):
    def loss(y_true,y_pred):
        p = y_pred[:,:,0:1]
        model_out = y_pred[:,:,1:]
        e_gt = tf_l2u(tf_u2l(y_true) - tf_u2l(p))
        prob_compensation = tf.squeeze((K.abs(e_gt - 128)/128.0)*K.log(256.0))
        alpha = e_gt - tf.math.floor(e_gt)
        alpha = tf.tile(alpha,[1,1,256])
        e_gt = tf.cast(e_gt,'int32')
        e_gt = tf.clip_by_value(e_gt,0,254) 
        interp_probab = (1 - alpha)*model_out + alpha*tf.roll(model_out,shift = -1,axis = -1)
        sparse_cel = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)(e_gt,interp_probab)
        loss_mod = sparse_cel + gamma*prob_compensation
        return loss_mod
    return loss

# Same as above, except a metric
def metric_oginterploss(y_true,y_pred):
    p = y_pred[:,:,0:1]
    model_out = y_pred[:,:,1:]
    e_gt = tf_l2u(tf_u2l(y_true) - tf_u2l(p))
    prob_compensation = tf.squeeze((K.abs(e_gt - 128)/128.0)*K.log(256.0))
    alpha = e_gt - tf.math.floor(e_gt)
    alpha = tf.tile(alpha,[1,1,256])
    e_gt = tf.cast(e_gt,'int32')
    e_gt = tf.clip_by_value(e_gt,0,254) 
    interp_probab = (1 - alpha)*model_out + alpha*tf.roll(model_out,shift = -1,axis = -1)
    sparse_cel = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)(e_gt,interp_probab)
    loss_mod = sparse_cel + prob_compensation
    return loss_mod

# Interpolated cross entropy loss metric
def metric_icel(y_true, y_pred):
    p = y_pred[:,:,0:1]
    model_out = y_pred[:,:,1:]
    e_gt = tf_l2u(tf_u2l(y_true) - tf_u2l(p))
    alpha = e_gt - tf.math.floor(e_gt)
    alpha = tf.tile(alpha,[1,1,256])
    e_gt = tf.cast(e_gt,'int32')
    e_gt = tf.clip_by_value(e_gt,0,254) #Check direction
    interp_probab = (1 - alpha)*model_out + alpha*tf.roll(model_out,shift = -1,axis = -1)
    sparse_cel = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)(e_gt,interp_probab)
    return sparse_cel

# Non-interpolated (rounded) cross entropy loss metric
def metric_cel(y_true, y_pred):
    p = y_pred[:,:,0:1]
    model_out = y_pred[:,:,1:]
    e_gt = tf_l2u(tf_u2l(y_true) - tf_u2l(p))
    e_gt = tf.round(e_gt)
    e_gt = tf.cast(e_gt,'int32')
    e_gt = tf.clip_by_value(e_gt,0,255) 
    sparse_cel = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)(e_gt,model_out)
    return sparse_cel

# Variance metric of the output excitation
def metric_exc_sd(y_true,y_pred):
    p = y_pred[:,:,0:1]
    e_gt = tf_l2u(tf_u2l(y_true) - tf_u2l(p))
    sd_egt = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)(e_gt,128)
    return sd_egt

# Scaled Linear Domain Loss (as a \mu law replacement)
def interp_sld(gamma = 1,smallG = 1,largeG = 1, lpc_reg = 1):
    def loss(y_true,y_pred):
        p = y_pred[:,:,0:1]
        G = y_pred[:,:,1:2]
        model_out = y_pred[:,:,2:]
        G = K.sqrt((1 + G)/(1 - G))
        # G = G*0 + 20
        reg_largeG = K.log(G)
        e_gt_sld = (tf_u2l(y_true) - tf_u2l(p))/G
        lpc_force = tf.reduce_mean(tf.square(e_gt_sld), axis=-1)
        reg_smallG = K.maximum(K.abs(e_gt_sld) - 127,0) # Check once
        # e_gt_mulaw = tf_l2u(tf_u2l(y_true) - tf_u2l(p))
        # prob_compensation = tf.squeeze((K.abs(e_gt_mulaw - 128)/128.0)*K.log(256.0))
        # prob_compensation = tf.squeeze((K.abs(e_gt_sld)/128.0)*K.log(256.0)) # Use old \mu law 
        e_gt_sld_noncentered = e_gt_sld + 128
        e_gt_sld_noncentered = tf.clip_by_value(e_gt_sld_noncentered,0,255) 
        alpha = e_gt_sld_noncentered - tf.math.floor(e_gt_sld_noncentered)
        alpha = tf.tile(alpha,[1,1,256])
        e_gt_sld_noncentered = tf.cast(e_gt_sld_noncentered,'int32')
        # e_gt_sld_noncentered = tf.clip_by_value(e_gt_sld_noncentered,0,254) 
        interp_probab = (1 - alpha)*model_out + alpha*tf.roll(model_out,shift = -1,axis = -1)
        sparse_cel = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)(e_gt_sld_noncentered,interp_probab)
        loss_mod = sparse_cel + largeG*reg_largeG[:,:,0] + smallG*reg_smallG[:,:,0] + lpc_reg*lpc_force
        return loss_mod
    return loss

def metric_reg_largeG(y_true, y_pred):
    G = y_pred[:,:,1:2]
    G = K.sqrt((1 + G)/(1 - G))
    reg_largeG = K.log(G)
    return reg_largeG

def metric_reg_smallG(y_true, y_pred):
    p = y_pred[:,:,0:1]
    G = y_pred[:,:,1:2]
    G = K.sqrt((1 + G)/(1 - G))
    e_gt_sld = (tf_u2l(y_true) - tf_u2l(p))/G
    reg_smallG = K.maximum(K.abs(e_gt_sld) - 128,0)
    return reg_smallG

def metric_probcompensation(y_true, y_pred):
    p = y_pred[:,:,0:1] 
    G = y_pred[:,:,1:2]
    G = K.sqrt((1 + G)/(1 - G))
    e_gt_sld = (tf_u2l(y_true) - tf_u2l(p))/G
    prob_compensation = tf.squeeze((K.abs(e_gt_sld)/128.0)*K.log(256.0))
    return prob_compensation

def metric_mulaw_icel(y_true, y_pred):
    p = y_pred[:,:,0:1] 
    model_out = y_pred[:,:,2:]
    e_gt = tf_l2u(tf_u2l(y_true) - tf_u2l(p))
    alpha = e_gt - tf.math.floor(e_gt)
    alpha = tf.tile(alpha,[1,1,256])
    e_gt = tf.cast(e_gt,'int32')
    e_gt = tf.clip_by_value(e_gt,0,254) #Check direction
    interp_probab = (1 - alpha)*model_out + alpha*tf.roll(model_out,shift = -1,axis = -1)
    sparse_cel = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)(e_gt,interp_probab)
    return sparse_cel

def metric_sld_icel(y_true,y_pred):
    p = y_pred[:,:,0:1]
    G = y_pred[:,:,1:2]
    model_out = y_pred[:,:,2:]
    G = K.sqrt((1 + G)/(1 - G))
    e_gt_sld = (tf_u2l(y_true) - tf_u2l(p))/G
    # prob_compensation = tf.squeeze((K.abs(e_gt_sld)/128.0)*K.log(256.0)) # Use old \mu law 
    e_gt_sld_noncentered = e_gt_sld + 128
    e_gt_sld_noncentered = tf.clip_by_value(e_gt_sld_noncentered,0,255) 
    alpha = e_gt_sld_noncentered - tf.math.floor(e_gt_sld_noncentered)
    alpha = tf.tile(alpha,[1,1,256])
    e_gt_sld_noncentered = tf.cast(e_gt_sld_noncentered,'int32')
    interp_probab = (1 - alpha)*model_out + alpha*tf.roll(model_out,shift = -1,axis = -1)
    sparse_cel = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)(e_gt_sld_noncentered,interp_probab)
    return sparse_cel

def metric_G(y_true,y_pred):
    G = y_pred[:,:,1:2]
    G = K.sqrt((1 + G)/(1 - G))
    return tf.reduce_mean(G, axis=-1)

def metric_lpcreg(y_true,y_pred):
    p = y_pred[:,:,0:1]
    G = y_pred[:,:,1:2]
    G = K.sqrt((1 + G)/(1 - G))
    # G = G*0 + 20
    e_gt_sld = (tf_u2l(y_true) - tf_u2l(p))/G
    lpc_force = tf.reduce_mean(tf.square(e_gt_sld), axis=-1)
    return lpc_force