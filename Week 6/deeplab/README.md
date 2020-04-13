# How Deeplab works

This document describe the steps necessary to run Deeplab with the Cityscapes dataset using
a Xception-65 pre-trained on ImageNet.


## Convert Cityscapes dataset to TFRecord
First of all, the dataset needs to be converted into the TFRecord format that deeplab uses. In order to do so, 2 tasks need to be performed:

1. Add the cityscapesScripts folder from https://github.com/mcordts/cityscapesScripts into the Cityscapes dataset folder.

2. Run the script to convert Cityscapes to TFRecord. The script is inside the datasets folder

```bash
bash convert_cityscapes.sh
```


## Training on Deeplab
Deeplab allows high customisation of the parameters of the network, that can be modified when running the training script (train.py). Following there is an example of how to run the training script:

```bash
python train.py \
    --logtostderr \
    --training_number_of_steps=5000 \
    --train_split="train" \
    --model_variant="xception_65" \
    --atrous_rates=6 \
    --atrous_rates=12 \
    --atrous_rates=18 \
    --output_stride=16 \
    --decoder_output_stride=4 \
    --train_crop_size='769,769' \
    --add_image_level_feature=True \
    --train_batch_size=1 \
    --dataset="cityscapes" \
    --tf_initial_checkpoint='./datasets/cityscapes/checkpoints/xception/model.ckpt' \
    --train_logdir='./datasets/cityscapes/exp/train_on_train_set/train/exp3' \
    --dataset_dir='./datasets/cityscapes/tfrecord'
```

A few important things:

1.  The tf_initial_checkpoint is the path to where the pre-trained network is stored. This pre-trained network can be download in https://github.com/mathildor/DeepLab-v3/blob/master/g3doc/model_zoo.md. In our case, we have used the pre-trained xception network on ImageNet (http://download.tensorflow.org/models/deeplabv3_xception_2018_01_04.tar.gz).
2.  The train_logdir is the path to where the checkpoints of the training process will be stored.
3.  The dataset_dir is where the TFRecords from the previous conversion have been stored.


## Evaluation on Deeplab


```bash
# From tensorflow/models/research/
python deeplab/eval.py \
    --logtostderr \
    --eval_split="val" \
    --model_variant="xception_65" \
    --atrous_rates=6 \
    --atrous_rates=12 \
    --atrous_rates=18 \
    --output_stride=16 \
    --decoder_output_stride=4 \
    --eval_crop_size=1025 \
    --eval_crop_size=2049 \
    --dataset="cityscapes" \
    --checkpoint_dir=${PATH_TO_CHECKPOINT} \
    --eval_logdir=${PATH_TO_EVAL_DIR} \
    --dataset_dir=${PATH_TO_DATASET}
```