#!/bin/bash
nohup python -u ../src/train.py -inputpath ~/datasets/asian-webface-lcnn-40/train.tfrecord  -n_gpus 2 -epoch_size 750106 --batch_size 64 --log_step 10000 --n_epochs 5 &

