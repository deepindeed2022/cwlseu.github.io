---
layout: post
title: "玩转Caffe之数据处理"
categories: [blog ]
tags: [Caffe, ]
description:  使用caffe框架进行实验过程中的一些心得
---

## 引入
最近实验中又跟caffe打交道，虽然caffe好用，但是要想让caffe启动训练起来，还真得费一番功夫。
数据处理，模型文件编写，预训练模型的选择等等。

## ImageNet的数据预处理

1. 常见image list

```shell
#!/bin/bash

root_dir=$HOME/data/VOCdevkit/
sub_dir=ImageSets/Main
bash_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for dataset in trainval test
do
  dst_file=$bash_dir/$dataset.txt
  if [ -f $dst_file ]
  then
    rm -f $dst_file
  fi
  for name in VOC2007 VOC2012
  do
    if [[ $dataset == "test" && $name == "VOC2012" ]]
    then
      continue
    fi
    echo "Create list for $name $dataset..."
    dataset_file=$root_dir/$name/$sub_dir/$dataset.txt

    img_file=$bash_dir/$dataset"_img.txt"
    cp $dataset_file $img_file
    sed -i "s/^/$name\/JPEGImages\//g" $img_file
    sed -i "s/$/.jpg/g" $img_file

    label_file=$bash_dir/$dataset"_label.txt"
    cp $dataset_file $label_file
    sed -i "s/^/$name\/Annotations\//g" $label_file
    sed -i "s/$/.xml/g" $label_file

    paste -d' ' $img_file $label_file >> $dst_file

    rm -f $label_file
    rm -f $img_file
  done

  # Generate image name and size infomation.
  if [ $dataset == "test" ]
  then
    $bash_dir/../../build/tools/get_image_size $root_dir $dst_file $bash_dir/$dataset"_name_size.txt"
  fi

  # Shuffle trainval file.
  if [ $dataset == "trainval" ]
  then
    rand_file=$dst_file.random
    cat $dst_file | perl -MList::Util=shuffle -e 'print shuffle(<STDIN>);' > $rand_file
    mv $rand_file $dst_file
  fi
done
```

2. 生成backend为leveldb或者lmdb

```shell
#!/usr/bin/env sh
# Create the imagenet lmdb inputs
# N.B. set the path to the imagenet train + val data dirs
set -e

DATA=$HOME/data/VOCdevkit
TOOLS=$HOME/caffe/build/tools

EXAMPLE=${DATA}/VOC0712/lmdb
TRAIN_DATA_ROOT=${DATA}/
VAL_DATA_ROOT=${DATA}/

# Set RESIZE=true to resize the images to 256x256. Leave as false if images have
# already been resized using another tool.
RESIZE=true
if $RESIZE; then
  RESIZE_HEIGHT=256
  RESIZE_WIDTH=256
else
  RESIZE_HEIGHT=0
  RESIZE_WIDTH=0
fi

echo "Creating train lmdb..."

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle \
    $TRAIN_DATA_ROOT \
    $DATA/trainval.txt \
    $EXAMPLE/voc0712_train_lmdb

echo "Creating val lmdb..."

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle \
    $VAL_DATA_ROOT \
    $DATA/test.txt \
    $EXAMPLE/voc0712_val_lmdb

echo "Done."
```

3. 在caffe中使用

```
layer {
  name: "data"
  type: "Data"
  top: "data"
  top: "label"
  data_param {
    source: "/home/limin/data/VOCdevkit/VOC0712/lmdb/voc0712_train_lmdb"
    #source: " /home/limin/data/VOCdevkit/VOC0712/voc0712_train_leveldb"
    mean_file: "/home/limin/data/VOCdevkit/VOC0712/voc0712_mean.binaryproto"
    batch_size: 16 
    crop_size: 227 
    # 数据类型，默认情况下为leveldb
    backend: LMDB
  }
  transform_param{
    mirror: true
  }
}
```
其中具体的参数需要参考caffe.proto文件进行查看，进行正确的配置

## 源码解析
### convert_imageset

## 问题
1. 如果一直在如下位置夯住，不继续运行了的话：
```
layer {
name: "conv5_1/linear"
type: "Convoluti
I0320 15:59:15.669935 20624 layer_factory.hpp:77] Creating layer data
I0320 15:59:15.670370 20624 net.cpp:100] Creating Layer data
I0320 15:59:15.670387 20624 net.cpp:408] data -> data
I0320 15:59:15.670454 20624 net.cpp:408] data -> label
```
可能是训练数据类型是对的，但是去取过程中出现了，这个时候就要检查是不是训练数据的使用的是测试数据的地址。我就是犯了
这么错误，找了好久终于找到了。

2. 进行模型funetune的时候，prototxt和.caffemodel一定要对应，否则真的会出现各种shape size不匹配的问题

3. 编写prototxt的时候要风格统一。不要layers和layer模式混用。

> 风格1: Layers开头，type为全部大写不带引号

```
layers {
  bottom: "fc7"
  top: "fc7"
  name: "drop7"
  type: DROPOUT
  dropout_param {
    dropout_ratio: 0.5
  }
}
```

> 风格2：layer开头，类型为首字母大写的字符串

```
layer {
  name: "conv1"
  type: "Convolution"
  bottom: "data"
  top: "conv1"
  param {
    lr_mult: 1
  }
  param {
    lr_mult: 2
  }
  convolution_param {
    num_output: 20
    kernel_size: 5
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
    }
  }
}
```

> 风格3：layers和layer嵌套类型

```
layers {
  layer {
    name: "conv2"
    type: "conv"
    num_output: 256
    group: 2
    kernelsize: 5
    weight_filler {
      type: "gaussian"
      std: 0.01
    }
    bias_filler {
      type: "constant"
      value: 1.
    }
    blobs_lr: 1.
    blobs_lr: 2.
    weight_decay: 1.
    weight_decay: 0.
  }
  bottom: "pad2"
  top: "conv2"
}
```

编写的时候保持风格统一就好。