---
layout: post
title: "SSD中的数据增强分析"
categories: [blog ]
tags: [深度学习, 物体检测, 数据增强]
description: 数据增强技术在CV研究中对于提高Performance是重要的研究话题。尤其是在物体检测方面，业界流行的方法中对具体方法之外，往往通过数据增强技术再次提高几个百分点。听着很诱人, 那么从SSD中的数据增强开始。
---

声明：本博客欢迎转发，但请保留原作者信息!
作者: 曹文龙
博客： <https://cwlseu.github.io/>

本文是Wei Liu在2016年的一篇成果. 采用VOC2007 Dataset, Nvidia Titan X上：

	mAP： 74.3% 
	59FPS
	使用数据增强技术可以达到77.2%

## SSD(Single Shot MultiBox Detector)

## 源代码分析
example/ssd/ssd_pascal.py

### 数据增强入口
```python
# Create train net.
# NOTE: Where the data from
net = caffe.NetSpec()
net.data, net.label = CreateAnnotatedDataLayer(train_data, batch_size=batch_size_per_device,
                                               train=True, output_label=True, label_map_file=label_map_file,
                                               transform_param=train_transform_param, batch_sampler=batch_sampler)


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

### 参数

```python
# // Sample a bbox in the normalized space [0, 1] with provided constraints.
# message Sampler {
#   // Minimum scale of the sampled bbox.
#   optional float min_scale = 1 [default = 1.];
#   // Maximum scale of the sampled bbox.
#   optional float max_scale = 2 [default = 1.];
#   // Minimum aspect ratio of the sampled bbox.
#   optional float min_aspect_ratio = 3 [default = 1.];
#   // Maximum aspect ratio of the sampled bbox.
#   optional float max_aspect_ratio = 4 [default = 1.];
# }
# // Constraints for selecting sampled bbox.
# message SampleConstraint {
#   // Minimum Jaccard overlap between sampled bbox and all bboxes in
#   // AnnotationGroup.
#   optional float min_jaccard_overlap = 1;
#   // Maximum Jaccard overlap between sampled bbox and all bboxes in
#   // AnnotationGroup.
#   optional float max_jaccard_overlap = 2;
#   // Minimum coverage of sampled bbox by all bboxes in AnnotationGroup.
#   optional float min_sample_coverage = 3;
#   // Maximum coverage of sampled bbox by all bboxes in AnnotationGroup.
#   optional float max_sample_coverage = 4;
#   // Minimum coverage of all bboxes in AnnotationGroup by sampled bbox.
#   optional float min_object_coverage = 5;
#   // Maximum coverage of all bboxes in AnnotationGroup by sampled bbox.
#   optional float max_object_coverage = 6;
# }
# // Sample a batch of bboxes with provided constraints.
# message BatchSampler {
#   // Use original image as the source for sampling.
#   optional bool use_original_image = 1 [default = true];
#   // Constraints for sampling bbox.
#   optional Sampler sampler = 2;
#   // Constraints for determining if a sampled bbox is positive or negative.
#   optional SampleConstraint sample_constraint = 3;
#   // If provided, break when found certain number of samples satisfing the
#   // sample_constraint.
#   optional uint32 max_sample = 4;
#   // Maximum number of trials for sampling to avoid infinite loop.
#   optional uint32 max_trials = 5 [default = 100];
# }

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

# // Message that stores parameters used to apply transformation
# // to the data layer's data
# message TransformationParameter {
#   // For data pre-processing, we can do simple scaling and subtracting the
#   // data mean, if provided. Note that the mean subtraction is always carried
#   // out before scaling.
#   optional float scale = 1 [default = 1];
#   // Specify if we want to randomly mirror data.
#   optional bool mirror = 2 [default = false];
#   // Specify if we would like to randomly crop an image.
#   optional uint32 crop_size = 3 [default = 0];
#   optional uint32 crop_h = 11 [default = 0];
#   optional uint32 crop_w = 12 [default = 0];

#   // mean_file and mean_value cannot be specified at the same time
#   optional string mean_file = 4;
#   // if specified can be repeated once (would substract it from all the channels)
#   // or can be repeated the same number of times as channels
#   // (would subtract them from the corresponding channel)
#   repeated float mean_value = 5;
#   // Force the decoded image to have 3 color channels.
#   optional bool force_color = 6 [default = false];
#   // Force the decoded image to have 1 color channels.
#   optional bool force_gray = 7 [default = false];
#   // Resize policy
#   optional ResizeParameter resize_param = 8;
#   // Noise policy
#   optional NoiseParameter noise_param = 9;
#   // Distortion policy
#   optional DistortionParameter distort_param = 13;
#   // Expand policy
#   optional ExpansionParameter expand_param = 14;
#   // Constraint for emitting the annotation after transformation.
#   optional EmitConstraint emit_constraint = 10;
# }
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

进入Annotated_data_layer.cpp

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
## 参考文献
1. [ssd源代码]<https://github.com/weiliu89/caffe.git>
