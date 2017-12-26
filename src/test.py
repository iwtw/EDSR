import tensorflow as tf
import numpy as np
import argparse
import os
import sys
import edsr
from tf_tools.save_images import *

args = None
def _parse_function( filename ):
    image_string = tf.read_file( filename )
    image_decoded =  tf.image.decode_image( image_string ) 
    image_converted = tf.image.convert_image_dtype( image_decoded ,  tf.float32 )
    image_converted.set_shape([None,None,3])
    return image_converted 

def build_graph(  ):
    filenames = tf.placeholder(tf.string,shape=[None])
    dataset = tf.data.TextLineDataset( filenames )
    dataset = dataset.map(_parse_function)
    dataset = dataset.batch(args.batch_size)

    iterator = dataset.make_initializable_iterator()
    I_LR = iterator.get_next()
    I_LR_split = tf.split( I_LR , args.n_gpus )


    I_SR_list = []
    for i in range(args.n_gpus):
        with tf.device( "/gpu:{}".format(i) ):
            I_LR_i = I_LR_split[i]
            if i == 0 :
                reuse = None
            else:
                reuse = True
            I_SR_i = edsr.edsr( I_LR_i , args.scale, dim = args.dim , reuse = reuse )
            I_SR_list.append(I_SR_i)
    
    I_SR = tf.concat(I_SR_list , axis = 0 )
    return  filenames , iterator , I_SR

        
def test(  filenames , iterator ,  I_SR):

    filename_list = open( args.input ).read().split("\n")
    if filename_list[-1] == "":
        filename_list.pop()
    saver = tf.train.Saver(var_list = tf.get_collection( tf.GraphKeys.GLOBAL_VARIABLES))
    config = tf.ConfigProto( allow_soft_placement = True )
    config.gpu_options.allow_growth = True
    with tf.Session( config = config ) as sess:
        assert os.path.exists( args.model_dir + "model.meta" ) , "model file doesn't exists"
        saver.restore( sess , args.model_dir + "model" )

        sess.run(iterator.initializer , feed_dict = {filenames:[args.input]})
        idx_batch = 0
        while True:
            try:
                out = sess.run( I_SR  )
                save_images(  filename_list[ args.batch_size * idx_batch : args.batch_size * (idx_batch+1) ] , np.array(out) , args.output + args.name + "/"  )
                idx_batch += 1
            except tf.errors.OutOfRangeError:
                break


def parse_args():
    parser = argparse.ArgumentParser( description = "test" )
    parser.add_argument("-input","-i")
    parser.add_argument("-output","-o",default="../test_output/")
    parser.add_argument('-n_gpus',type=int,default = 1)
    parser.add_argument('--batch_size',type=int,default=64)
    parser.add_argument('--dim',type=int,default=256)
    parser.add_argument('--upsample_method',default="subpixel")
    parser.add_argument('--height',type=int,default=112)
    parser.add_argument('--width',type=int , default=96)
    parser.add_argument('--scale',type=int,default=4)
    parser.add_argument('--n_epochs',type=int,default=20)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    args.name = "edsr_dim{}_{}_scale{}_epoch{}".format(args.dim , args.upsample_method , args.scale , args.n_epochs )
    args.model_dir = "../model/"+args.name + "/"  
    if args.output[-1] != "/":
        args.output += "/"
    
    filenames , iterator , I_SR = build_graph()
    test( filenames , iterator , I_SR )
