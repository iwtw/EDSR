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



def preprocessing_function( image_string )  :
    image_decoded = tf.image.decode_image( image_string , 3 )
    image_float = tf.cast( image_decoded , tf.float32  )
    image_reg = 1.0/127.5 * image_float - 1.0 
    return image_reg

def _parse_function( example_proto ):
    feature = {
                "label": tf.FixedLenFeature( () , tf.int32) ,
                "img_raw" : tf.FixedLenFeature( () , tf.string) 
            }
    parsed_feature = tf.parse_single_example( example_proto , feature )
    image_string = parsed_feature["img_raw"]
    preprocessed_image = preprocessing_function( image_string  )
    return parsed_feature['label'] , preprocessed_image


def build_and_train( args ):
    def build_dataset(filenames):
        dataset = tf.data.TFRecordDataset( train_filenames )
        dataset.map(_parse_function )
        dataset.shuffle( buffer_size = 10000 )
        dataset.batch( args.batch_size )
        return dataset
    
    train_dataset = build_dataset( args.train_input )
    val_dataset = build_dataset( args.val_input )
    val_dataset = val_dataset.repeat( -1 )

    train_iterator = train_dataset.make_initializable_iterator()
    val_iterator = val_dataset.make_initializable_iterator()
    iterators = {}
    iterators["train"] = train_iterator
    iterators["val"] = val_iterator


    handle = tf.placeholder( tf.string , shape=[])
    iterator = tf.data.Iterator.from_string_handle( handle , train_dataset.output_types , train_dataset.output_shapes )


    label , I_LR = iterator.get_next()
    I_LR_split = tf.split( I_LR , args.n_gpus ) 
    I_BI = tf.image.resize_bicubic( I_LR , ( args.height , args.width ) )

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
    
    #####train

    epoch_step = tf.placeholder( tf.int32 )
    global_step = tf.Variable( 0 , trainable = False )
    boundaries = [ decay_period * i * args.n_its_per_epoch for i in range(int(args.n_epochs / decay_period ))  ]
    learning_rate = tf.train.exponential_decay( args.learning_rate , global_step = epoch_step , decay_step  = 1  , decay_rate = 0.1 )
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


        it = global_step.eval

        merged_summary = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter("../log/{}/train".format( args.name ),sess.graph)
        val_writer = tf.summary.FileWriter("../log/{}/val".format( args.name)  , sess.graph )

        train_handle , val_handle = sess.run( [ iterators["train"].string_handle()   , iterators["val"].string_handle ] )
        sess.run(iterators["val"].initializer , feed_dict = {filenames:[args.val_input]})

        for epoch in range(args.n_epochs):
            sess.run(iterators["train"].initializer )
            while True:
                try:
                    _ , train_log = sess.run([train_op,merged_summary] , feed_dict ={ handle:train_handle  , epoch_step:epoch })
                except tf.errors.OutOfRangeError:
                    break

            def save_log():
                label_ , val_log = sess.run([ label ,  merged_summary] ,feed_dict = { handle:val_handle })
                train_writer.add_summary( train_log , epoch )
                val_writer.add_summary( val_log ,epoch )
                train_writer.flush()
                val_writer.flush()
                saver.save( sess , args.model_dir+"/model" )
                i_lr , i_sr = sess.run( [ i_LR , I_SR ] , feed_dict = "val_handle")
                i_bi = skimage.transform.resize( i_lr , (args.height , args.width) )
                i_hr = np.zeros( (args.batch_size , args.height , args.width , 3 ) )
                for i in range(len(label_)):
                    lr_path = args.label_to_path.get( label_[i] )
                    lr_path_split = lr_path.split("/")
                    hr_path = ""
                    for j in range( len(lr_path_split) - 3  ):
                        hr_path += lr_path_split[j]
                        hr_path += "/"
                    dir_name = lr_path_split[:-3]
                    dir_name_split = dir_name.split("_")
                    for k in range( len( dir_name_split) - 1 ):
                        hr_path += dir_name_split[j]
                    hr_path += lr_path_split[-2] + "/"
                    hr_path += lr_path_split[-1]
                    i_hr[i] = skimgae.io.imread(hr_path)

                save_images.save_training_images("../training_output/"+args.name,[i_hr,i_bi,i_sr], epoch )

            save_log()




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
    parser.add_argument("-val_input",help="val tfrecord filename")
    parser.add_argument('-n_gpus',type=int)
    parser.add_argument("-label_to_path",type=str,help="json filename storing label to path mapping")
    parser.add_argument('--batch_size',type=int,default=64)
    parser.add_argument('--height',type=int,default=112)
    parser.add_argument('--width',type=int , default=96)
    parser.add_argument('--dim',type=int,default=256)
    parser.add_argument('--scale',type=int,default=4)
    parser.add_argument('--upsample_method',default="subpixel")
    parser.add_argument('--learning_rate',type=float,default=1e-6)
    parser.add_argument('--n_epochs',type=int,default=20)
    args = parser.parse_args()

    args.name = "edsr_dim{}_{}_scale{}_epoch{}".format(args.dim , args.upsample_method , args.scale , args.n_epochs )
    args.model_dir = "../model/"+args.name   
    args.n_its_per_epoch = int( args.epoch_size / args.batch_size ) + ( args.epoch_size % args.batch_size != 0  )
    args.label_to_path = json.load( open(args.label_to_path , "r") )
    return args

def main(_):
    args = parse_args()

    init_dir(args)
    
    build_and_train(args)
    
if __name__ == "__main__":
    tf.app.run(main)

