import tensorflow as tf
import edsr
import argparse

SR = edsr.edsr


def parse_args():
    parser = arg.parse.ArgumentTparser( description = "train" )
    parser.add_argument('--inputdir',help='input tfrecord dir')
    parser.add_argument('--outputdir','-o')
    parser.add_argument('--model','-m',help ='model path')
    parser.add_argument('--name','-n',help='checkpoint path')
    parser.add_argument('--n_gpu',type=int)
    parser.add_argument('--batch_size','-b',default=16)
    parser.add_argument('--width',type=int , default=96)
    parser.add_argument('--height',type=int,default=112)
    parser.add_argument('--dim',type=int,default=256)
    parser.add_argument('--upsample_method',default)
    return parser.parse_args()

def build_graph( args ):
    datadir = tf.
    file_queue = tf.train.string_input_producer(
    imgs , labels  = 

def train():

def main(_):
    args = parse_args()

    

    train()
    
if __name__ == "__main__":
    tf.run(main)

