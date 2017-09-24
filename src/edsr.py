import tensorflow as tf
import numpy as np
import utils

def block( inputs , kernel_size , stride , activation_fn = None , regularization_scale = 0.0 , scaling_factor = 0.1 , scope = "Block"):
    "EDSR block conv -> relu -> conv "
    "inputs : tensor of format 'NHWC' "
    "returns: tensor of same shape as inputs "
    with tf.variable_scope(scope) :
        dim =  inputs.get_shape()[-1].value
        outputs = utils.conv2d( inputs , dim , kernel_size , stride ,he_init =  True , activation_fn = None , regularization_scale = regularization_scale  )
        outputs = tf.nn.relu( outputs   )
        outputs = utils.conv2d( outputs , dim , kernel_size , stride , he_init =  False , activation_fn = None , regularization_scale = regularization_scale  )
        outputs = inputs + scaling_factor * outputs
    return outputs

def edsr( inputs , scale  ,  dim = 256 , block_length = 32  , upsample_method = "subpixel" ,  regularization_scale = 0.0 , reuse = None ):
    "EDSR model"
    "inputs : tensor of format 'NHWC' "
    "returns : tensor I^SR "
    print("EDSR------------------------------------")
    with tf.variable_scope("EDSR" , reuse = reuse ) :
        conv1 = utils.conv2d( inputs , dim , 3 , 1 , regularization_scale = regularization_scale  )
        outputs = conv1
        for i in range(block_length):
            outputs = block( outputs , 3 , 1 , tf.nn.relu , regularization_scale= regularization_scale , scope = "Block{}".format(i) )
        outputs = utils.conv2d( outputs , dim , 3 , 1 , regularization_scale = regularization_scale  )
        outputs += conv1
        outputs = utils.upsample( outputs ,  scale = scale , dim = dim , upsample_method = upsample_method ,  regularization_scale = regularization_scale   )
        outputs = utils.conv2d( outputs , 3 , 3 , 1 , regularization_scale = regularization_scale ) 
    return outputs

