import tensorflow as tf
import numpy as np
import time as time
import tensorflow.contrib.layers as layers

def conv2d( inputs , outputs_dim , kernel_size ,   stride ,   padding = "SAME" , he_init = False , activation_fn = None , regularization_scale = 0.0 , name = None ): 
    C = inputs.get_shape()[-1].value
    fan_in = C * kernel_size**2
    fan_out = C * kernel_size**2 / stride**2
    avg_fan = (fan_in + fan_out) / 2
    if he_init:
        var = 2.0/avg_fan
    else :
        var = 1.0/avg_fan
    # var = (b - a)**2 / 12 , b==-a ,  (zero mean)
    upper_bound = np.sqrt( 12.0*var ) * 0.5 
    weights_initializer = tf.random_uniform_initializer( -upper_bound , upper_bound , seed = None , dtype = tf.float32 )
    weights_regularizer = layers.l2_regularizer( scale = regularization_scale )
    return layers.conv2d( inputs = inputs , num_outputs = outputs_dim , kernel_size = kernel_size , stride =  stride, padding = "SAME"  , activation_fn = activation_fn , weights_initializer = weights_initializer  , name = name , weights_regularizer = weights_regularizer )

def fully_connected( inputs , outputs_dim ,  he_init = False , activation_fn = None , regularization_scale = 0.0 , name = None ):
    x = layers.flatten( inputs )
    fan_in = x.get_shape()[-1].value
    fan_out = ( C + outputsdim ) / 2
    avg_fan = ( fan_in + fan_out ) / 2 
    if he_init:
        var = 2.0/avg_fan
    else:
        var = 1.0/avg_fan
    # var = (b - a)**2 / 12 , b==-a ,  (zero mean)
    upper_bound = np.sqrt( 12.0 * var ) *0.5
    weights_initializer = tf.random_uniform_initializer( -upper_bound , upper_bound , seed = None , dtype = tf.float32 )
    weights_regularizer = layers.l2_regularizer( scale = regularization_scale )
    return layers.fully_connected( x , outputs_dim , weights_initializer =  weights_initializer , activation_fn = activation_fn  , name = name ,   weights_regularizer = weights_regularizer )

def conv2d_transpose( inputs , outputs_dim , kernel_size , stride , padding = "SAME" , he_init = False , activation_fn = None , regularization_scale = 0.0 , name = None  ):
    C = inputs.get_shape()[-1].value
    fan_in = C * kernel_size**2 / stride**2
    fan_out = C * kernel_size**2 
    avg_fan = ( fan_in + fan_out ) / 2 
    if he_init:
        var = 2.0/avg_fan
    else :
        var = 1.0/avg_fan
    # var = ( b - a )**2 /12 , b==-a , (zero mean)
    upper_bound = np.sqrt( 12.0 * var ) *0.5
    weights_initializer = tf.random_uniform_initializer( -upper_bound , upper_bound , seed = None , dtype = tf.float32 )
    weights_regularizer = layers.l2_regularizer( scale = regularization_scale )
    return layers.conv2d_transpose( inputs , outputs_dim , kernel_size = kernel_size , stride = stride , padding = padding ,  weights_initializer = weights_initializer ,  activation_fn = activation_fn , name = name, weights_regularizer = weights_regularizer ):


def upsample( inputs , scale , dim    , upsample_method = "subpixel" ,  activation_fn = None , regularization_scale = 0.0 , name = None ):
    "upsample layer"
    act = activation_fn
    if act == None:
        act = tf.identity
    with tf.variable_scope(name):
        if upsample_method == "subpixel":
            if scale == 2 :
                ouputs = utils.conv2d(  inputs ,  dim , 3 , 1 , he_init = (activation_fn == tf.nn.relu ) , activation_fn = activation_fn ,  regularization_scale = regularization_scale , name = "conv1") 
                ouputs = tf.depth_to_space( outputs , 2 )
                outputs = act( outputs )
            elif scale == 3 :
                outputs = utils.conv2d( inputs , dim , 3 , 1 ,  he_init = (activation_fn == tf.nn.relu ) , activation_fn = activation_fn , regularization_scale = regularization_scale , name = "conv1")
                outputs = tf.depth_to_space( ouputs , 3 )
                outputs = act( outputs )
            elif scale == 4 :
                ouputs = utils.conv2d(  inputs ,  dim , 3 , 1 , regularization_scale = regularization_scale , name = "conv1") 
                ouputs = tf.depth_to_space( outputs , 2 )
                ouputs = utils.conv2d(  ouputs ,  dim , 3 , 1 , he_init = (activation_fn == tf.nn.relu ) , activation_fn = activation_fn , regularization_scale = regularization_scale , name = "conv2") 
                ouputs = tf.depth_to_space( outputs , 2 )
                outputs = act( outputs )
        elif upsample_method == "conv_transpose":
            if scale == 2 :
                outputs = utils.conv2d_transpose( inputs , dim , 3 , 2 , he_init = (activation_fn == tf.nn.relu ) , activation_fn = activation_fn , regularization_scale = regularization_scale , name = "conv_transpose1"
                outputs = act( outputs )
            elif scale == 3:
                outputs = utils.conv2d_transpose( inputs , dim , 3 , 3 , he_init = (activation_fn == tf.nn.relu ) , activation_fn = activation_fn , regularization_scale = regularization_scale , name = "conv_transpose1"
                outputs = act( outputs )
            elif scale == 4:
                outputs = utils.conv2d_transpose( inputs , dim , 3 , 2 , regularization_scale = regularization_scale , name = "conv_transpose1"
                outputs = utils.conv2d_transpose( outputs , dim , 3 , 2 , he_init = (activation_fn == tf.nn.relu ) , activation_fn = activation_fn , regularization_scale = regularization_scale , name = "conv_transpose2"
                outputs = act( outputs )
                
    return outputs
