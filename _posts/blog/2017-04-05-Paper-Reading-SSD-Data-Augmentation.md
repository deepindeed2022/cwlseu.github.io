---
layout: post
title: "深度学习：SSD中的数据增强分析"
categories: [project]
tags: [深度学习, detection, CV算法]
description: 数据增强技术在CV研究中对于提高Performance是重要的研究话题。尤其是在物体检测方面，业界流行的方法中对具体方法之外，往往通过数据增强技术再次提高几个百分点。
---

{:toc}

## SSD(Single Shot MultiBox Detector)
本文是Wei Liu在2016年的一篇成果. 采用VOC2007 Dataset, Nvidia Titan X上：

> mAP： 74.3% 
> 59FPS
> 使用数据增强技术可以达到77.2%

## 模型关键词

使用前向反馈CNN网络，产生固定数目的bounding box， 然后再这些bounding box中进行打分。

non-maximum suppression step
Non-Maximum Suppression就是根据score和box的坐标信息，从中找到置信度比较高的bounding box。
- 首先，根据score进行排序，把score最大的bounding box拿出来。
- 计算其余bounding box与这个box的IoU，然后去除IoU大于设定的阈值的bounding box。
- 重复上面的过程，直至候选bounding box为空。
说白了就是要在一堆矩阵里面找出一些**局部最大值**，所以要把和这些局部最大值所代表矩阵IoU比较大的去除掉，这样就能得到一些权值很大，而且IoU又比较小的bounding box。

## 源代码分析

### anno_type_

`has_anno_type_ = anno_datum.has_type() || anno_data_param.has_anno_type();` 最后的结果是什么？其中`anno_data_param.has_anno_type()` 结果是false, 关键看anno_datum中有没有了。这个里面有没有要去看你运行`create_data.sh`的时候对数据进行了什么操作。在本文中是对其中写了`AnnotatedDatum_AnnotationType_BBOX`类型

```cpp
    ······
    else if (anno_type == "detection") {
      // 数据转化过程中写入的类型
      labelname = root_folder + boost::get<std::string>(lines[line_id].second);
      status = ReadRichImageToAnnotatedDatum(filename, labelname, resize_height,
          resize_width, min_dim, max_dim, is_color, enc, type, label_type,
          name_to_label, &anno_datum);
      anno_datum.set_type(AnnotatedDatum_AnnotationType_BBOX);
    }
    ······
```

因此此处`has_anno_type`就是true。

### 过程概述

for i in batch_size:
  1. 先对图片Expand操作或者Distort操作进行处理
    - 首先从数据队里中获取img，但是不进行删除。接着对这个image进行拓展\distort操作
    - Expand过程中是在随机生成一个拓增后的大小图片expand_img
    - 采用平均值填充图片
    - 将图片向拓增后的图片进行copy
    - 该操作结束之后，就生成了expand_img，后面在expand_img基础上进行操作
  2. 生成采样 
    - 入口`GenerateBatchSamples(*expand_datum, batch_samplers_, &sampled_bboxes);`
    - 其中需要注意的是生成了若干个sampled_bboxes,但是只是从中随机选择了一个进行裁剪;否则直接使用原来的expand后的的数据
  3. 对sampled_datum进行resize
    - 调用Data_transformer进行转化
    - 其中包括对AnnotationData中的Datum进行转化
    - Annotation的转化
      + 其中包括resize和重新映射等操作
      + 需要重新映射标注中Bounding Box的大小
      + expaned_image中的annotation进行annotation转化之后返回数据类型
   
    vector<AnnotationGroup>
        |-- AnnotationGroup
               |-- group_label
               |-- Annotation多个
                      |-- bbox
                      |-- instance_id
                  
  4. 对采样后的数据重新编码到blob中
  5. 将处理后的数据push到batch数据流中
  `reader_.free().push(const_cast<AnnotatedDatum*>(&anno_datum));`
endfor

6. 重新处理annotation数据
  最后的top_label数据shape为：1 x 1 x num_boxs x 8

### 数据增强入口

```python
# Create train net.
# NOTE: Where the data from
net = caffe.NetSpec()
net.data, net.label = CreateAnnotatedDataLayer(train_data, batch_size=batch_size_per_device,train=True, output_label=True, label_map_file=label_map_file,transform_param=train_transform_param, batch_sampler=batch_sampler)
def CreateAnnotatedDataLayer(source, batch_size=32, backend=P.Data.LMDB,
        output_label=True, train=True, label_map_file='', anno_type=None,
        transform_param={}, batch_sampler=[{}]):
    if train:
        kwargs = {
                'include': dict(phase=caffe_pb2.Phase.Value('TRAIN')),
                'transform_param': transform_param,
                }
    else:
        kwargs = {
                'include': dict(phase=caffe_pb2.Phase.Value('TEST')),
                'transform_param': transform_param,
                }
    ntop = 1
    if output_label:
        ntop = 2
    annotated_data_param = {
        'label_map_file': label_map_file,
        'batch_sampler': batch_sampler,
        }
    if anno_type is not None:
        annotated_data_param.update({'anno_type': anno_type})
    return L.AnnotatedData(name="data", annotated_data_param=annotated_data_param,
        data_param=dict(batch_size=batch_size, backend=backend, source=source),
        ntop=ntop, **kwargs)
```

### 参数说明

#### 一个sampler的参数说明
    // Sample a bbox in the normalized space [0, 1] with provided constraints.
    message Sampler {
    // 最大最小scale数
    optional float min_scale = 1 [default = 1.];
    optional float max_scale = 2 [default = 1.];
    // 最大最小采样长宽比，真实的长宽比在这两个数中间取值
    optional float min_aspect_ratio = 3 [default = 1.];
    optional float max_aspect_ratio = 4 [default = 1.];
    }

#### 对于选择的sample_box的限制条件
    // Constraints for selecting sampled bbox.
    message SampleConstraint {
      // Minimum Jaccard overlap between sampled bbox and all bboxes in
      // AnnotationGroup.
      optional float min_jaccard_overlap = 1;
      // Maximum Jaccard overlap between sampled bbox and all bboxes in
      // AnnotationGroup.
      optional float max_jaccard_overlap = 2;
      // Minimum coverage of sampled bbox by all bboxes in AnnotationGroup.
      optional float min_sample_coverage = 3;
      // Maximum coverage of sampled bbox by all bboxes in AnnotationGroup.
      optional float max_sample_coverage = 4;
      // Minimum coverage of all bboxes in AnnotationGroup by sampled bbox.
      optional float min_object_coverage = 5;
      // Maximum coverage of all bboxes in AnnotationGroup by sampled bbox.
      optional float max_object_coverage = 6;
    }
我们们往往只用max_jaccard_overlap

#### 对于一个batch进行采样的参数设置
    // Sample a batch of bboxes with provided constraints.
    message BatchSampler {
      // 是否使用原来的图片
      optional bool use_original_image = 1 [default = true];
      // sampler的参数
      optional Sampler sampler = 2;
      // 对于采样box的限制条件，决定一个采样数据positive or negative
      optional SampleConstraint sample_constraint = 3;
      // 当采样总数满足条件时，直接结束
      optional uint32 max_sample = 4;
      // 为了避免死循环，采样最大try的次数.
      optional uint32 max_trials = 5 [default = 100];
    }

#### 转存datalayer数据的参数

    message TransformationParameter {
      // 对于数据预处理，我们可以仅仅进行scaling和减掉预先提供的平均值。
      // 需要注意的是在scaling之前要先减掉平均值
      optional float scale = 1 [default = 1];
      // 是否随机镜像操作
      optional bool mirror = 2 [default = false];
      // 是否随机crop操作
      optional uint32 crop_size = 3 [default = 0];
      optional uint32 crop_h = 11 [default = 0];
      optional uint32 crop_w = 12 [default = 0];
      // 提供mean_file的路径，但是不能和mean_value同时提供
      // if specified can be repeated once (would substract it from all the 
      // channels) or can be repeated the same number of times as channels
      // (would subtract them from the corresponding channel)
      optional string mean_file = 4;
      repeated float mean_value = 5;
      // Force the decoded image to have 3 color channels.
      optional bool force_color = 6 [default = false];
      // Force the decoded image to have 1 color channels.
      optional bool force_gray = 7 [default = false];
      // Resize policy
      optional ResizeParameter resize_param = 8;
      // Noise policy
      optional NoiseParameter noise_param = 9;
      // Distortion policy
      optional DistortionParameter distort_param = 13;
      // Expand policy
      optional ExpansionParameter expand_param = 14;
      // Constraint for emitting the annotation after transformation.
      optional EmitConstraint emit_constraint = 10;
    }

#### SSD中的数据转换和采样参数设置

```python
# sample data parameter
batch_sampler = [
    {
      # use_original_image : true,
        'sampler': {
        },
        'max_trials': 1,
        'max_sample': 1,
    },
    {
        'sampler': {
            'min_scale': 0.3,
            'max_scale': 1.0,
            'min_aspect_ratio': 0.5,
            'max_aspect_ratio': 2.0,
        },
        'sample_constraint': {
            'min_jaccard_overlap': 0.1,
        },
        'max_trials': 50,
        'max_sample': 1,
    },
    ......
]


## 考虑添加inv数据
train_transform_param = {
    'mirror': True,
    'mean_value': [104, 117, 123],
    'resize_param': {
        'prob': 1,
        'resize_mode': P.Resize.WARP,
        'height': resize_height,
        'width': resize_width,
        'interp_mode': [
            P.Resize.LINEAR,
            P.Resize.AREA,
            P.Resize.NEAREST,
            P.Resize.CUBIC,
            P.Resize.LANCZOS4,
        ],
    },
    'distort_param': {
        'brightness_prob': 0.5,
        'brightness_delta': 32,
        'contrast_prob': 0.5,
        'contrast_lower': 0.5,
        'contrast_upper': 1.5,
        'hue_prob': 0.5,
        'hue_delta': 18,
        'saturation_prob': 0.5,
        'saturation_lower': 0.5,
        'saturation_upper': 1.5,
        'random_order_prob': 0.0,
    },
    # This param related with the size of expand image
    'expand_param': {
        'prob': 0.5,
        'max_expand_ratio': 4.0,
    },
    'emit_constraint': {
        'emit_type': caffe_pb2.EmitConstraint.CENTER,
    }
}
```

进入Annotated_data_layer.cpp 进入cpp调用阶段。

### ExpandImage

首先根据max_expand_radio和读取的数据进行生成具有expand_radio的expand_datum和一个expand_bbox

```cpp
template<typename Dtype>
void DataTransformer<Dtype>::ExpandImage(const AnnotatedDatum& anno_datum,
                                         AnnotatedDatum* expanded_anno_datum) {

  const ExpansionParameter& expand_param = param_.expand_param();
  const float expand_prob = expand_param.prob();
  float prob;
  caffe_rng_uniform(1, 0.f, 1.f, &prob);
  if (prob > expand_prob) {
    expanded_anno_datum->CopyFrom(anno_datum);
    return;
  }
  // 如果进行expand图片的话，获取expand的大小
  const float max_expand_ratio = expand_param.max_expand_ratio();
  if (fabs(max_expand_ratio - 1.) < 1e-2) {
    expanded_anno_datum->CopyFrom(anno_datum);
    return;
  }
  float expand_ratio;
  caffe_rng_uniform(1, 1.f, max_expand_ratio, &expand_ratio);
  // Expand the datum.
  NormalizedBBox expand_bbox;
  ExpandImage(anno_datum.datum(), expand_ratio, &expand_bbox,
              expanded_anno_datum->mutable_datum());
  expanded_anno_datum->set_type(anno_datum.type());

  // Transform the annotation according to crop_bbox.
  const bool do_resize = false;
  const bool do_mirror = false;
  TransformAnnotation(anno_datum, do_resize, expand_bbox, do_mirror,
                      expanded_anno_datum->mutable_annotation_group());
}
```

### 生成采样

#### 入口

```cpp
    if (batch_samplers_.size() > 0) {
        // Generate sampled bboxes from expand_datum.
        vector<NormalizedBBox> sampled_bboxes;
        //进入关键函数
        // 
        GenerateBatchSamples(*expand_datum, batch_samplers_, &sampled_bboxes);
        if (sampled_bboxes.size() > 0) {
          // Randomly pick a sampled bbox and crop the expand_datum.
          int rand_idx = caffe_rng_rand() % sampled_bboxes.size();
          sampled_datum = new AnnotatedDatum();
          this->data_transformer_->CropImage(*expand_datum,
                                             sampled_bboxes[rand_idx],
                                             sampled_datum);
          has_sampled = true;
        } else {
          sampled_datum = expand_datum;
        }
      } else {
        sampled_datum = expand_datum;
      }
```
  

#### 具体调用过程

```cpp

  // // An extension of Datum which contains "rich" annotations.
  // message AnnotatedDatum {
  //   enum AnnotationType {
  //     BBOX = 0;
  //   }
  //   optional Datum datum = 1;
  //   // If there are "rich" annotations, specify the type of annotation.
  //   // Currently it only supports bounding box.
  //   // If there are no "rich" annotations, use label in datum instead.
  //   optional AnnotationType type = 2;
  //   // Each group contains annotation for a particular class.
  //   repeated AnnotationGroup annotation_group = 3;
  // }

  void GenerateBatchSamples(const AnnotatedDatum& anno_datum,
                            const vector<BatchSampler>& batch_samplers,
                            vector<NormalizedBBox>* sampled_bboxes) {
    sampled_bboxes->clear();
    vector<NormalizedBBox> object_bboxes;
    //将所有的objectbox重新存储在object_bboxes
    GroupObjectBBoxes(anno_datum, &object_bboxes);
    for (int i = 0; i < batch_samplers.size(); ++i) {
      if (batch_samplers[i].use_original_image()) {
        NormalizedBBox unit_bbox;
        unit_bbox.set_xmin(0);
        unit_bbox.set_ymin(0);
        unit_bbox.set_xmax(1);
        unit_bbox.set_ymax(1);
        GenerateSamples(unit_bbox, object_bboxes, batch_samplers[i],
                        sampled_bboxes);
      }
    }
  }
  void GenerateSamples(const NormalizedBBox& source_bbox,
                     const vector<NormalizedBBox>& object_bboxes,
                     const BatchSampler& batch_sampler,
                     vector<NormalizedBBox>* sampled_bboxes) {
    int found = 0;
    for (int i = 0; i < batch_sampler.max_trials(); ++i) 
    {
      // 每次最多采样batch_sampler.max_sample()
      if (batch_sampler.has_max_sample() &&
          found >= batch_sampler.max_sample()) {
        break;
      }
      // Generate sampled_bbox in the normalized space [0, 1].
      NormalizedBBox sampled_bbox;
      SampleBBox(batch_sampler.sampler(), &sampled_bbox);
      // Transform the sampled_bbox w.r.t. source_bbox.
      LocateBBox(source_bbox, sampled_bbox, &sampled_bbox);
      // Determine if the sampled bbox is positive or negative by the constraint.
      if (SatisfySampleConstraint(sampled_bbox, object_bboxes,
                                  batch_sampler.sample_constraint())) {
        ++found;
        sampled_bboxes->push_back(sampled_bbox);
      }
    }
  }
  void SampleBBox(const Sampler& sampler, NormalizedBBox* sampled_bbox) {
    // Get random scale.
    // CHECK_GE(sampler.max_scale(), sampler.min_scale());
    // CHECK_GT(sampler.min_scale(), 0.);
    // CHECK_LE(sampler.max_scale(), 1.);
    float scale;
    caffe_rng_uniform(1, sampler.min_scale(), sampler.max_scale(), &scale);

    // Get random aspect ratio.
    // CHECK_GE(sampler.max_aspect_ratio(), sampler.min_aspect_ratio());
    // CHECK_GT(sampler.min_aspect_ratio(), 0.);
    // CHECK_LT(sampler.max_aspect_ratio(), FLT_MAX);
    float aspect_ratio;
    float min_aspect_ratio = std::max<float>(sampler.min_aspect_ratio(),
                                             std::pow(scale, 2.));
    float max_aspect_ratio = std::min<float>(sampler.max_aspect_ratio(),
                                             1 / std::pow(scale, 2.));
    caffe_rng_uniform(1, min_aspect_ratio, max_aspect_ratio, &aspect_ratio);

    // Figure out bbox dimension.
    float bbox_width = scale * sqrt(aspect_ratio);
    float bbox_height = scale / sqrt(aspect_ratio);

    // Figure out top left coordinates.
    float w_off, h_off;
    caffe_rng_uniform(1, 0.f, 1 - bbox_width, &w_off);
    caffe_rng_uniform(1, 0.f, 1 - bbox_height, &h_off);

    sampled_bbox->set_xmin(w_off);
    sampled_bbox->set_ymin(h_off);
    sampled_bbox->set_xmax(w_off + bbox_width);
    sampled_bbox->set_ymax(h_off + bbox_height);
  }
```


```cpp
  cv::Rect bbox_roi(w_off, h_off, img_width, img_height);
  img.copyTo((*expand_img)(bbox_roi));
```

example/ssd/ssd_pascal.py

## 回顾调用路线

```cpp
// This function is called on prefetch thread
template<typename Dtype>
void AnnotatedDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  ......

  // Store transformed annotation.
  map<int, vector<AnnotationGroup> > all_anno;
  int num_bboxes = 0;

  for (int item_id = 0; item_id < batch_size; ++item_id) {
    timer.Start();
    // get a anno_datum
    AnnotatedDatum& anno_datum = *(reader_.full().pop("Waiting for data"));
    read_time += timer.MicroSeconds();
    timer.Start();
    AnnotatedDatum distort_datum;
    AnnotatedDatum* expand_datum = NULL;
    // 对图片进行光照处理
    if (transform_param.has_distort_param()) {
      distort_datum.CopyFrom(anno_datum);
      this->data_transformer_->DistortImage(anno_datum.datum(),
                                            distort_datum.mutable_datum());
      if (transform_param.has_expand_param()) {
        // 对图片进行数量拓展，其中实现了对annotation的转化等等操作
        expand_datum = new AnnotatedDatum();
        this->data_transformer_->ExpandImage(distort_datum, expand_datum);
      } else {
        // 对图片进行扭曲变形
        expand_datum = &distort_datum;
      }
    } else {
      if (transform_param.has_expand_param()) {
        expand_datum = new AnnotatedDatum();
        this->data_transformer_->ExpandImage(anno_datum, expand_datum);
      } else {
        expand_datum = &anno_datum;
      }
    }

    AnnotatedDatum* sampled_datum = NULL;
    bool has_sampled = false;
    if (batch_samplers_.size() > 0) {
      // Generate sampled bboxes from expand_datum.
      vector<NormalizedBBox> sampled_bboxes;
      //进入关键函数
      // 
      GenerateBatchSamples(*expand_datum, batch_samplers_, &sampled_bboxes);
      if (sampled_bboxes.size() > 0) {
        // Randomly pick a sampled bbox and crop the expand_datum.
        int rand_idx = caffe_rng_rand() % sampled_bboxes.size();
        sampled_datum = new AnnotatedDatum();
        this->data_transformer_->CropImage(*expand_datum,
                                           sampled_bboxes[rand_idx],
                                           sampled_datum);
        has_sampled = true;
      } else {
        sampled_datum = expand_datum;
      }
    } else {
      sampled_datum = expand_datum;
    }

    ......

    // Apply data transformations (mirror, scale, crop...)
    int offset = batch->data_.offset(item_id);
    this->transformed_data_.set_cpu_data(top_data + offset);
    vector<AnnotationGroup> transformed_anno_vec;
    if (this->output_labels_) {
      if (has_anno_type_) {
        // Make sure all data have same annotation type.
        CHECK(sampled_datum->has_type()) << "Some datum misses AnnotationType.";
        if (anno_data_param.has_anno_type()) {
          sampled_datum->set_type(anno_type_);
        } else {
          CHECK_EQ(anno_type_, sampled_datum->type()) <<
              "Different AnnotationType.";
        }
        // Transform datum and annotation_group at the same time
        transformed_anno_vec.clear();
        // 对sample后的数据进行转化
        this->data_transformer_->Transform(*sampled_datum,
                                           &(this->transformed_data_),
                                           &transformed_anno_vec);
        if (anno_type_ == AnnotatedDatum_AnnotationType_BBOX) {
          // Count the number of bboxes.
          for (int g = 0; g < transformed_anno_vec.size(); ++g) {
            num_bboxes += transformed_anno_vec[g].annotation_size();
          }
        } else {
          LOG(FATAL) << "Unknown annotation type.";
        }
        all_anno[item_id] = transformed_anno_vec;
      } else {
        this->data_transformer_->Transform(sampled_datum->datum(),
                                           &(this->transformed_data_));
        // Otherwise, store the label from datum.
        CHECK(sampled_datum->datum().has_label()) << "Cannot find any label.";
        top_label[item_id] = sampled_datum->datum().label();
      }
    } else {
      this->data_transformer_->Transform(sampled_datum->datum(),
                                         &(this->transformed_data_));
    }
    // clear memory
    if (has_sampled) {
      delete sampled_datum;
    }
    if (transform_param.has_expand_param()) {
      delete expand_datum;
    }
    trans_time += timer.MicroSeconds();

    reader_.free().push(const_cast<AnnotatedDatum*>(&anno_datum));
  }

  // Store "rich" annotation if needed.
  if (this->output_labels_ && has_anno_type_) {
    vector<int> label_shape(4);
    if (anno_type_ == AnnotatedDatum_AnnotationType_BBOX) {
      label_shape[0] = 1;
      label_shape[1] = 1;
      label_shape[3] = 8;
      if (num_bboxes == 0) {
        // Store all -1 in the label.
        label_shape[2] = 1;
        batch->label_.Reshape(label_shape);
        caffe_set<Dtype>(8, -1, batch->label_.mutable_cpu_data());
      } else {
        // Reshape the label and store the annotation.
        label_shape[2] = num_bboxes;
        batch->label_.Reshape(label_shape);
        top_label = batch->label_.mutable_cpu_data();
        int idx = 0;
        for (int item_id = 0; item_id < batch_size; ++item_id) {
          const vector<AnnotationGroup>& anno_vec = all_anno[item_id];
          for (int g = 0; g < anno_vec.size(); ++g) {
            const AnnotationGroup& anno_group = anno_vec[g];
            for (int a = 0; a < anno_group.annotation_size(); ++a) {
              const Annotation& anno = anno_group.annotation(a);
              const NormalizedBBox& bbox = anno.bbox();
              top_label[idx++] = item_id;
              top_label[idx++] = anno_group.group_label();
              top_label[idx++] = anno.instance_id();
              top_label[idx++] = bbox.xmin();
              top_label[idx++] = bbox.ymin();
              top_label[idx++] = bbox.xmax();
              top_label[idx++] = bbox.ymax();
              top_label[idx++] = bbox.difficult();
            }
          }
        }
      }
    } else {
      LOG(FATAL) << "Unknown annotation type.";
    }
  }
}
```

## 参考文献

1. [ssd源代码]<https://github.com/weiliu89/caffe.git>
