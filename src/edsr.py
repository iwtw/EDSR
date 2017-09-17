import tensorflow as tf
import numpy as np
import utils

def block( inputs , filter_size , stride , activation_fn = None , regularization_scale = 0.0 , scaling_factor = 0.1 , name = "block"):
    "EDSR block conv -> relu -> conv "
    "inputs : tensor of format 'NHWC' "
    "returns: tensor of same shape as inputs "
    with tf.variable_scope(name):
        dim =  inputs.get_shape()[-1].value
        outputs = utils.conv2d( inputs , dim , filter_size , stride , True , None , regularization_scale , name = "conv1" )
        outputs = tf.nn.relu( outputs , name = "relu1"  )
        outputs = utils.conv2d( outputs , dim , filter_size , stride ,  False , None , regularization_scale , name = "conv2" )
        outputs = inputs + scaling_factor * outputs
    return outputs

def edsr( inputs , scale  ,  dim = 256 , block_length = 32  , upsample_method = "subpixel" ,  regularization_scale = 0.0  ):
    "EDSR model"
    "inputs : tensor of format 'NHWC' "
    "returns : tensor I^SR "
    with tf.variable_scope("EDSR"):
        conv1 = utils.conv2d( inputs , dim , 3 , 1 , regularization_scale = regularization_scale , name = "conv1" )
        outputs = conv1
        for i in range(block_length):
            outputs = block( outputs , 3 , 1 , tf.nn.relu , regularization_scale= regularization_scale , name = "block{}".format(i+1))
        outputs = utils.conv2d( ouputs , dim , 3 , 1 , regularization_scale = regularization_scale , name = "conv2" )
        outputs += conv1
        outputs = utils.upsample( ouputs ,  scale = scale , dim = dim , upsample_method = upsample_method ,  regularization_scale = regularization_scale , name = "upsample"  )
    return outputs

