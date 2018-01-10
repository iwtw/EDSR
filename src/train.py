import tensorflow as tf
import argparse
import os
import sys
sys.path.append("../src")
import edsr
from tf_tools import save_images
from tf_tools import data_input
import skimage.transform
import skimage.io
import json
import time
from utils import *
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}


#load the mean of dataset
with open( "mean.json" , "r" ) as json_file:
    dataset_mean_dict = json.load(  json_file )
dataset_mean = []
dataset_mean.append( float( dataset_mean_dict.get( "R" )))
dataset_mean.append( float( dataset_mean_dict.get( "G" )))
dataset_mean.append( float( dataset_mean_dict.get( "B" )))
dataset_mean = np.array( dataset_mean )



def _filter_function(n_gpus):
    def f(x,y):
        a = tf.equal( tf.mod( tf.shape(x)[0] , n_gpus ) , 0 )  
        b = tf.equal( tf.mod( tf.shape(y)[0] , n_gpus ) , 0 )  
        return tf.logical_and(a,b)
    return f
def build_and_train( args ):
    def build_dataset(filenames):
        dataset = ( tf.data.TFRecordDataset( filenames )
            .map( parse_function_for_train(args) , num_parallel_calls = 8 )
            .shuffle( buffer_size = 10000 )
            .batch( args.batch_size   )
            .filter(  _filter_function(args.n_gpus))
            )
        return dataset
    
    train_dataset = build_dataset( args.train_input )
    val_dataset = build_dataset( args.val_input )
    val_dataset = val_dataset.repeat( -1 )

    train_iterator = train_dataset.make_initializable_iterator()
    val_iterator = val_dataset.make_initializable_iterator()


    handle = tf.placeholder( tf.string , shape=[])
    iterator = tf.data.Iterator.from_string_handle( handle , train_dataset.output_types , train_dataset.output_shapes )


    I_LR , I_HR = iterator.get_next()
    I_LR_split = tf.split( I_LR , args.n_gpus ) 

    losses ,  I_SR_list  =  [] , []
    I_HR_split = tf.split(I_HR , args.n_gpus )
    for i in range(args.n_gpus):
        with tf.device( "/gpu:{}".format(i) ):
            I_LR_i = I_LR_split[i]
            I_HR_i = I_HR_split[i]
            if i == 0 :
                reuse = None
            else:
                reuse = True
            I_LR_i = preprocess( I_LR_i , dataset_mean )
            I_SR_i = edsr.edsr( I_LR_i , args.scale, dim = args.dim , n_blocks = args.n_blocks ,  reuse = reuse )
            I_SR_i = deprocess( I_SR_i , dataset_mean )
            I_SR_list.append(I_SR_i)
    
    I_SR = tf.concat(I_SR_list , axis = 0 )
    #L1 reconstruct loss
    loss = 1.0/args.batch_size *tf.reduce_sum( tf.abs( I_HR - I_SR )  )

    def PSNR( a , b , MAX):
        mse = tf.losses.mean_squared_error ( a , b )
        return 10.0 * tf.log( MAX **2 / mse)/tf.log(10.0)
    psnr = PSNR( I_SR , I_HR , 255.0 )
    tf.summary.scalar("PSNR" , psnr )
    tf.summary.scalar("loss",loss)
    

    #####train

    epoch_step = tf.Variable( 0 , trainable = False )
    epoch_step_inc = tf.assign_add( epoch_step , 1 )
    tf.summary.scalar("epoch",epoch_step)

    global_step = tf.Variable( 0 , trainable = False )
    learning_rate = tf.train.exponential_decay( args.learning_rate , global_step = epoch_step , decay_steps  = 1  , decay_rate = args.decay_rate )
    train_op = tf.train.AdamOptimizer( learning_rate = learning_rate ).minimize( loss , var_list = tf.get_collection( tf.GraphKeys.TRAINABLE_VARIABLES ) , global_step = global_step)
    saver = tf.train.Saver(var_list = tf.get_collection( tf.GraphKeys.GLOBAL_VARIABLES))

    
    config = tf.ConfigProto( allow_soft_placement = True )
    config.gpu_options.allow_growth = True

    with tf.Session( config = config ) as sess:
        if not os.path.exists( args.model_dir):
            os.mkdir( args.model_dir )
        if os.path.exists( args.model_dir + "/model.meta" ):
            saver.restore( sess , args.model_dir + "/model" )
        else :    
            sess.run( tf.global_variables_initializer() )


        merged_summary = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter("../log/{}/train".format( args.name ),sess.graph)
        val_writer = tf.summary.FileWriter("../log/{}/val".format( args.name)  , sess.graph )

        train_handle , val_handle = sess.run( [ train_iterator.string_handle()   , val_iterator.string_handle() ] )

        sess.run(val_iterator.initializer )
        #for epoch in range(args.n_epochs):
        prev_time = time.time()
        while epoch_step.eval() < args.n_epochs:
            sess.run(train_iterator.initializer )
            while True:
                try:
                    sess.run(train_op, feed_dict ={ handle:train_handle })
                    if global_step.eval() % args.log_step == 0 :
                        temp_time = time.time()
                        print("     epoch %d it %.0e , %.1f iters/s"%(epoch_step.eval(),global_step.eval(),args.log_step/(temp_time-prev_time)))
                        prev_time = temp_time
                        train_log = sess.run( merged_summary , feed_dict ={handle:train_handle })
                        train_writer.add_summary( train_log , global_step.eval() )
                        train_writer.flush()
                except tf.errors.OutOfRangeError:
                    sess.run( epoch_step_inc )
                    break

            def validate():
                val_log = sess.run( merged_summary ,feed_dict = { handle:val_handle })
                val_writer.add_summary( val_log ,global_step.eval() )
                val_writer.flush()
                saver.save( sess , args.model_dir+"/model" )
                i_lr , i_sr , i_hr = sess.run( [ I_LR , I_SR , I_HR ] , feed_dict = { handle : val_handle } )
                i_sr = np.clip( i_sr , 0 , 255 )
                i_sr = np.round(i_sr).astype(np.uint8)
                i_hr = np.round(i_hr).astype(np.uint8)
                def resize_images( i_lr  ):
                    i_bi = []
                    for i in range( len(i_lr)):
                        #bicubic
                        i_bi.append( skimage.transform.resize( i_lr[i] , (args.height , args.width) , order = 3 , clip = True , preserve_range = True )  )
                    return np.array(i_bi,dtype = np.uint8)
                i_bi = resize_images( i_lr )

                save_images.save_training_images([i_hr,i_bi,i_sr], epoch_step.eval(), "../training_output/"+args.name )

            #validate at each epoch end
            validate()


def init_dir(args):
    def mkdir(path):
        if not os.path.exists(path):
            os.mkdir(path)
    mkdir("../training_output") 
    mkdir("../model")
    mkdir("../log")

    mkdir("../training_output/"+args.name)

def parse_args():
    parser = argparse.ArgumentParser( description = "train" )
    parser.add_argument('-train_input',help='train tfrecord filename')
    parser.add_argument('-val_input',help='val tfrecord filename')
    parser.add_argument('-n_gpus',type=int)
    parser.add_argument('--log_step',type=int,default=100)
    parser.add_argument('--batch_size',type=int,default=64)
    parser.add_argument('--height',type=int,default=112)
    parser.add_argument('--width',type=int , default=96)
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

def main(_):
    args = parse_args()

    init_dir(args)
    
    build_and_train(args)
    
if __name__ == "__main__":
    tf.app.run(main)

