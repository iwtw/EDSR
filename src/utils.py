import numpy as np
import tensorflow as tf

    

def preprocess(images , dataset_mean):
    images -= np.reshape( dataset_mean , (1,1,1,3) )
    return images
def deprocess(images , dataset_mean):
    images += np.reshape( dataset_mean , (1,1,1,3))
    return images

def augment_image( image, flip_flag ,rotate_radian )  :
    image = tf.cond( flip_flag,
        lambda:tf.image.random_flip_left_right( image ) ,
        lambda:image
        )
    image = tf.contrib.image.rotate( image , rotate_radian , interpolation ="BILINEAR")
    return image

def decode_image( image_string  ):
    image = tf.image.decode_image( image_string , 3 )
    image.set_shape([None , None , 3])
    image = tf.cast( image, tf.float32  )
    return image
def parse_function_for_train(args ):
    def parser(example_proto):
        feature = {
                    "image_LR" : tf.FixedLenFeature( () , tf.string) ,
                    "image_HR" : tf.FixedLenFeature( () , tf.string) 
                }
        parsed_feature = tf.parse_single_example( example_proto , feature )
        image_LR_string = parsed_feature["image_LR"]
        image_HR_string = parsed_feature["image_HR"]

        image_LR = decode_image( image_LR_string )
        image_HR = decode_image( image_HR_string )
        #50% percent to flip the pair of images
        flip_flag = False
        if args.flip:
            flip_flag = tf.greater( tf.random_uniform( (1,1), 0,1 )[0,0] , 0.5)
        #random rotate 0,90,180,270 degree
        if args.rotate:
            rotate_radian =  tf.floor( tf.random_uniform( (1,1), 0,4 )[0,0]  )*90*np.pi/180
        else:
            rotate_radian = 0
        image_LR = augment_image( image_LR , flip_flag , rotate_radian )
        image_HR = augment_image( image_HR , flip_flag , rotate_radian )
        return image_LR , image_HR
    return parser

def parse_function_for_test(args  ):
    def parser( filename):
        image_string = tf.read_file( filename)
        image = decode_image( image_string  )
        return image
    return parser 
