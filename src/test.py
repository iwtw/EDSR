import tensorflow as tf
import numpy as np
import argparse
import os
import sys
import edsr
from tf_tools.save_images import *
from utils import *
import json

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

#load the mean of dataset
with open( "mean.json" , "r" ) as json_file:
    dataset_mean_dict = json.load(  json_file )
dataset_mean = []
dataset_mean.append( float( dataset_mean_dict.get( "R" )))
dataset_mean.append( float( dataset_mean_dict.get( "G" )))
dataset_mean.append( float( dataset_mean_dict.get( "B" )))
dataset_mean = np.array( dataset_mean )

def T(x):
    t_list = []
    shape = tf.cast(tf.shape( x) , tf.float32)
    for i in range( 2):
        #rotate x of ( (i//2) * 90) degree
  #      t = tf.contrib.image.rotate( x , (i//2) * np.pi * 90 / 180 ) 
        t = x 
        if i%2==0:
            t = tf.contrib.image.transform( t , [-1,0,shape[2]-1,0,1,0,0,0] )
        t_list.append( t )
    return tf.concat( t_list , axis = 0 )
        
def T_inv(x):
    t_list = []
    shape = tf.cast( tf.shape(x) , tf.int32)
    y = tf.reshape(x , tf.stack([2 , shape[0]//2 , shape[1] , shape[2] , shape[3] ], axis = 0 ))
    for i in range(2):
        #t = tf.contrib.image.rotate( y[i] , (i//2) * np.pi * (-90) / 180 )
        t = y[i]
        if i%2 == 0 :
            #horizontal flip
            t = tf.contrib.image.transform( t , [-1,0,tf.cast(shape[2],tf.float32)-1,0,1,0,0,0] )
        t_list.append(t)
    y = tf.stack( t_list , axis = 0 )
    return tf.reduce_mean( y , axis = 0 ) 

def build_and_test( args  ):
    filenames = tf.placeholder(tf.string,shape=[None])
    dataset = tf.data.TextLineDataset( filenames )
    dataset = dataset.map(parse_function_for_test(args))
    dataset = dataset.batch(args.batch_size)

    iterator = dataset.make_initializable_iterator()
    I_LR = iterator.get_next()
    I_LR_split = tf.split( I_LR , args.n_gpus )


    I_SR_list , I_SR_plus_list = [] , []
    reuse = None
    for i in range(args.n_gpus):
        with tf.device( "/gpu:{}".format(i) ):
            I_LR_i = I_LR_split[i]
            I_LR_i = preprocess( I_LR_i  , dataset_mean)
            
            I_SR_i = edsr.edsr( I_LR_i , args.scale, dim = args.dim , n_blocks = args.n_blocks , upsample_method = args.upsample_method ,  reuse = reuse )
            reuse = True
            I_SR_i = deprocess( I_SR_i , dataset_mean )
            I_SR_i = tf.clip_by_value( I_SR_i , 0 ,255 )
            I_SR_i = tf.cast( I_SR_i , tf.uint8)

            I_LR_plus_i = T(I_LR_i)
            I_SR_plus_i = edsr.edsr( I_LR_plus_i , args.scale, dim = args.dim , n_blocks = args.n_blocks , upsample_method = args.upsample_method ,  reuse = reuse )
            I_SR_plus_i = T_inv( I_SR_plus_i )
            I_SR_plus_i = deprocess( I_SR_plus_i , dataset_mean )
            I_SR_plus_i = tf.clip_by_value( I_SR_plus_i , 0 , 255 )
            I_SR_plus_i = tf.cast( I_SR_plus_i , tf.uint8 )

            I_SR_list.append(I_SR_i)
            I_SR_plus_list.append(I_SR_plus_i)
    
    I_SR = tf.concat(I_SR_list , axis = 0 )
    I_SR_plus = tf.concat(I_SR_plus_list , axis = 0 )
    ##########
    ###test###

    filename_list = open( args.test_input ).read().split("\n")
    if filename_list[-1] == "":
        filename_list.pop()
    config = tf.ConfigProto( allow_soft_placement = True )
    config.gpu_options.allow_growth = True
    with tf.Session( config = config ) as sess:
        assert os.path.exists( args.model_dir + "/" + "model.meta" ) , "model file %s doesn't exists"%(args.model_dir + "/" + "model.meta" )
        saver = tf.train.Saver(var_list = tf.get_collection( tf.GraphKeys.GLOBAL_VARIABLES))
        saver.restore( sess , args.model_dir + "/" + "model" )

        sess.run(iterator.initializer , feed_dict = {filenames:[args.test_input]})
        idx_batch = 0
        while True:
            try:
                out , out_plus = sess.run( [ I_SR , I_SR_plus]  )
                print( out.shape , out_plus.shape)
                save_images(  filename_list[ args.batch_size * idx_batch : args.batch_size * (idx_batch+1) ] , np.array(out) , args.output + "/" + args.name + "/"  )
                save_images(  filename_list[ args.batch_size * idx_batch : args.batch_size * (idx_batch+1) ] , np.array(out_plus) , args.output + "/" + args.name + "_plus" + "/"  )
                idx_batch += 1
            except tf.errors.OutOfRangeError:
                break

def parse_args():
    parser = argparse.ArgumentParser( description = "test" )
    parser.add_argument('-test_input',help="test input of low resolution , txt filename")
    parser.add_argument('-train_input',help='train tfrecord filename')
    parser.add_argument("-output","-o",help="test output dir name" ,type=str, default="../test_output")
    parser.add_argument('-n_gpus',type=int)
    parser.add_argument('--batch_size',type=int,default=64)
    parser.add_argument('--dim',type=int,default=256)
    parser.add_argument('--n_blocks',type=int,default=64)
    parser.add_argument('--scale',type=int,default=4)
    parser.add_argument('--upsample_method',default='subpixel')
    parser.add_argument('--learning_rate',type=float,default=1e-3)
    parser.add_argument('--decay_rate',type=float,default=0.5)
    parser.add_argument('--n_epochs',type=int,default=20)
    parser.add_argument('--flip',type=bool,default=True)
    parser.add_argument('--rotate',type=bool,default=True)
    args = parser.parse_args()
    args.name = "EDSR__dataset_%s__dim%d__n_blocks%d__%s__scale%d__epoch%d__lr%.0e__dr%.1f__rotate%d__flip%d"%(args.train_input.split("/")[-1].split(".")[-2] , args.dim , args.n_blocks , args.upsample_method , args.scale , args.n_epochs , args.learning_rate , args.decay_rate , args.rotate , args.flip )
    args.model_dir = "../model/"+args.name   
    return args

if __name__ == "__main__":
    args = parse_args()
    
    build_and_test(args)
