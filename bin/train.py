import tensorflow as tf
import argparse
import os
import sys
sys.path.append("../src")
import edsr
from tf_tools import save_images
from tf_tools import data_input


def parse_args():
    parser = argparse.ArgumentParser( description = "train" )
    parser.add_argument('-inputdir',help='input tfrecord dir')
#    parser.add_argument('-model',help='model dir')
    parser.add_argument('-n_gpus',type=int)
    parser.add_argument('-epoch_size',type=int)
    parser.add_argument('--batch_size','-b',default=16)
    parser.add_argument('--width',type=int , default=96)
    parser.add_argument('--height',type=int,default=112)
    parser.add_argument('--dim',type=int,default=256)
    parser.add_argument('--scale',type=int,default=4)
    parser.add_argument('--upsample_method',default="subpixel")
    parser.add_argument('--learning_rate',type=float,default=1e-6)
    parser.add_argument('--n_epochs',type=int,default=20)
    parser.add_argument('--log_step',type=int,default=10000)
    return parser.parse_args()

def build_graph( args ):
    datadir = tf.train.match_filenames_once(args.inputdir)
    file_queue = tf.train.string_input_producer(datadir,shuffle = True )
    I_HR , labels = data_input.get_batch(file_queue , (args.height,args.width) , args.batch_size , 8 , 5 , is_training = True )
    
    losses   =  []
    I_HR_split = tf.split(I_HR , args.n_gpus )
    for i in range(n_gpus):
        with tf.device( "gpu:/{}".format(i) ):
            I_HR_i = I_HR_split[i]
            I_LR_i = tf.image.resize_bicubic( I_HR_i , ( args.height/args.scale , args.width/args.scale ))
            I_SR_i = edsr.edsr( I_LR , args.scale, dim = args.dim )
            loss_i = tf.losses.absolute_difference( I_SR_i , I_HR_i )

            losses.append(loss_i)
            I_LR_list.append(I_LR_i)
            I_SR_list.append(I_SR_i)
    
    I_LR = tf.concat(I_LR_i , axis = 0 )
    I_SR = tf.concat(I_SR_i , axis = 0 )

    loss = tf.add_n( losses ) / args.n_gpus
    tf.summary.scalar("loss",loss)
    
    return loss , I_HR , I_LR , I_SR

    
def train( args , loss , I_HR , I_LR , I_SR ):
    global_step = tf.Variable( 0 , trainable = False )
    decay_period = 2 
    boundaries = [ decay_period * i * args.epoch_size for i in range(int(args.n_epochs / decay_period ))  ]
    LR = args.learning_rate 
    lrs = [ LR/2**i  for i in range(int(args.n_epochs / decay_period ))]
    train_op = tf.train.AdamOptimizer(learning_rate =lr).minimize( loss , var_list = tf.GraphKeys.TRAINABLE_VARIABLES,global_step = global_step)
    saver = tf.train.Saver(var_list = tf.get_collection( tf.GraphKeys.GLOBAL_VARIABLES))
    with tf.Session() as sess:
        if not os.path.exists( args.model):
            os.mkdir( args.model )
        if os.path.exists( args.model + "/model" ):
            saver.restore( sess , args.model + "model" )
        else :    
            sess.run( tf.global_variables_initializer())

        sess.run( tf.local_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners( sess )
        it = global_step.eval

        merged_summary = tf.summary.merge_all()
        writer = tf.summary.FileWRiter("../log/{}".format(NAME),sess.graph)
        while it() < args.epoch_size * args.n_epochs:
            loss , log = sess.run([loss , merged_summary])
            if it() % args.log_step == args.log_step -1 or it() == 0  :
                i_hr , i_lr , i_sr = sess.run( I_HR , I_LR , I_SR )
                writer.add_summary( log , it() )
                saver.save( sess , args.model+"/model" )
                save_images([i_hr,i_lr,i_sr],it() , it() / epoch_size + (it() % epoch_size==0) )
            sess.run(train_op)

        coord.request_stop()
        coord.join(threads)


def init_dir():
    def mkdir(path):
        if not os.path.exists(path):
            os.mkdir(path)
    mkdir("../training_output") 
    mkdir("../model")
    mkdir("../log")

def main(_):
    args = parse_args()
    args.model = "../model/edsr_dim{}_{}_scale{}".format(args.dim , args.upsample_method , args.scale)
    init_dir()

    
    loss , I_HR , I_LR , I_SR = build_graph(args)
    train( args , loss , I_HR , I_LR , I_SR )
    
if __name__ == "__main__":
    tf.app.run(main)

