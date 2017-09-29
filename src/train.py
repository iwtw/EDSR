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
    parser.add_argument('--batch_size',type=int,default=64)
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
    file_queue = tf.train.string_input_producer([args.inputdir] )
    I_HR , labels = data_input.get_batch(file_queue , (args.height,args.width) , args.batch_size , n_threads = 4 , min_after_dequeue = 5 , is_training = True )
    #img , _ = data_input.parse_single_data(file_queue)
    #with tf.Session() as sess:
    #    save_images.save_images( [img.eval] , 0, 0 )
    #assert 1==2
    I_LR = tf.image.resize_bicubic( I_HR , (int( args.height/args.scale) , int(args.width/args.scale) ))
    I_LR_split = tf.split( I_LR , args.n_gpus ) 
    I_BI = tf.image.resize_bicubic( I_LR , ( args.height , args.width) )

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
            I_SR_i = edsr.edsr( I_LR_i , args.scale, dim = args.dim , reuse = reuse )
            loss_i = tf.losses.absolute_difference( I_SR_i , I_HR_i ) 

            losses.append(loss_i)
            I_SR_list.append(I_SR_i)
    
    I_SR = tf.concat(I_SR_list , axis = 0 )

    loss = tf.add_n( losses ) / args.batch_size
    tf.summary.scalar("loss",loss)
    
    return loss , I_HR , I_BI , I_SR

    
def train( args , loss , I_HR , I_BI , I_SR ):

    global_step = tf.Variable( 0 , trainable = False )
    decay_period = 2 
    boundaries = [ decay_period * i * args.n_its_per_epoch for i in range(int(args.n_epochs / decay_period ))  ]
    LR = args.learning_rate 
    lrs = [ LR/2**i  for i in range(int(args.n_epochs / decay_period ))]
    lr = tf.train.piecewise_constant( global_step , boundaries , lrs )
    train_op = tf.train.AdamOptimizer(learning_rate =lr).minimize( loss , var_list = tf.get_collection( tf.GraphKeys.TRAINABLE_VARIABLES ) ,global_step = global_step)
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

        sess.run( tf.local_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners( sess )
        it = global_step.eval

        merged_summary = tf.summary.merge_all()
        writer = tf.summary.FileWriter("../log/{}".format( args.name ),sess.graph)
        while it() < args.n_its_per_epoch * args.n_epochs:
            loss_ , log = sess.run([loss , merged_summary])
            if it() % args.log_step == args.log_step -1 or it() < 10 :
                i_hr , i_bi , i_sr = sess.run( [I_HR , I_BI , I_SR ])
                writer.add_summary( log , it() )
                writer.flush()
                saver.save( sess , args.model_dir+"/model" )
                epoch = int ( it() / args.n_its_per_epoch ) + ( it() % args.n_its_per_epoch!=0)
                save_images.save_images("../training_output/"+args.name,[i_hr,i_bi,i_sr],it() , epoch )
               # print("step:{}".format(it()))
            sess.run(train_op)

        coord.request_stop()
        coord.join(threads)


def init_dir(args):
    def mkdir(path):
        if not os.path.exists(path):
            os.mkdir(path)
    mkdir("../training_output") 
    mkdir("../model")
    mkdir("../log")

    mkdir("../training_output/"+args.name)


def main(_):
    args = parse_args()
    args.name = "edsr_dim{}_{}_scale{}_epoch{}".format(args.dim , args.upsample_method , args.scale , args.n_epochs )
    args.model_dir = "../model/"+args.name   
    args.n_its_per_epoch = int( args.epoch_size / args.batch_size ) + ( args.epoch_size % args.batch_size != 0  )

    init_dir(args)
    
    loss , I_HR , I_LR , I_SR = build_graph(args)
    train( args , loss , I_HR , I_LR , I_SR )
    
if __name__ == "__main__":
    tf.app.run(main)

