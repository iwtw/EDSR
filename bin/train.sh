#!/bin/bash
python -u ../src/train.py -train_input ~/datasets/idcard/id_10w/id_10w_HR+LR_28x24.tfrecord  -val_input ~/datasets/idcard/id_10w/id_10w_HR+LR_28x24.tfrecord -n_gpus 2 -label_to_path ~/datasets/idcard/id_10w/id_10w_28x24_label.json --height 112 --width 96 --dim 64 --n_blocks 16 --scale 4 --batch_size 128  --learning_rate 1e-3 --decay_rate 0.1 --n_epochs 10 

