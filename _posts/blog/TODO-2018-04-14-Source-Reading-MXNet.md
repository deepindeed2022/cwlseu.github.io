---
layout: post
title: "深度学习：mxnet源代码阅读笔记"
categories: [blog ]
tags: [MXNet, 深度学习]
description: mxnet源代码阅读笔记
---

{:toc} 

## 快速安装[^1],[^2]

```shell
# 安装 build tools 和git
sudo apt-get update
sudo apt-get install -y build-essential git
# 安装OpenBLAS
sudo apt-get install -y libopenblas-dev liblapack-dev
# 安装OpenCV
sudo apt-get install -y libopencv-dev
# 安装Python bounding
sudo apt-get install -y python-dev python-setuptools python-numpy python-pip

sudo apt-get install graphviz
pip install graphviz

git clone --recursive https://github.com/apache/incubator-mxnet.git mxnet --branch 0.12.1
cd mxnet
make -j $(nproc) USE_OPENCV=1 USE_BLAS=openblas
```


当然，如果GPU版本的，需要先安装CUDA和cuDNN，然后添加`export LD_LIBRARY_PATH=/usr/local/cuda/lib64/:$LD_LIBRARY_PATH`

## 干脆配置一个notebook服务器[^4]

```shell
## 首先从这里下载http://anaconda.com/downloads.html anaconda进行安装
sudo apt-get install python-pip
sudo apt-get install ipython
pip install jupyter
# 
```

## 源代码阅读

### 发展过程

ImageNet的classification ->

代码主要路径`python/mxnet`

`operator.py`中含有大量的`ctypes`的使用案例，可以学习一番。学习的时候需要对照着`include/mxnet/c_api.h`中

## operator.py
* PythonOp(object)
	* NumpyOp(PythonOp)
	* NDArrayOp(PythonOp)
* CustomOp(object)
* CustomProp(object)
[如何自定义一个Operator呢](https://mxnet.incubator.apache.org/how_to/new_op.html)
1. pyhton中创建自定义Operator `mxnet.operator.CustomOp`

```python

class Softmax(mx.operator.CustomOp):
    def forward(self, is_train, req, in_data, out_data, aux):
		# takes a list of input and a list of output NDArrays
        x = in_data[0].asnumpy()
        y = np.exp(x - x.max(axis=1).reshape((x.shape[0], 1)))
        y /= y.sum(axis=1).reshape((x.shape[0], 1))
		# e used CustomOp.assign to assign the resulting array y to out_data[0]
        self.assign(out_data[0], req[0], mx.nd.array(y))
	def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
		l = in_data[1].asnumpy().ravel().astype(np.int)
		y = out_data[0].asnumpy()
		y[np.arange(l.shape[0]), l] -= 1.0
		self.assign(in_grad[0], req[0], mx.nd.array(y))

@mx.operator.register("softmax")
class SoftmaxProp(mx.operator.CustomOpProp):
	def __init__(self):
		# because softmax is a loss layer and you don’t need gradient input from preceding layers
		super(SoftmaxProp, self).__init__(need_top_grad=False)
	
	def list_arguments(self):
		return ['data', 'label']

	def list_outputs(self):
		return ['output']
	# Next, provide infer_shape to declare the shape of the output/weight and check the consistency of the input shapes:
	def infer_shape(self, in_shape):
		# The first axis of an input/output tensor corresponds to different examples within the batch. The label is a set of integers, one for each data entry, and the output has the same shape as the input
		data_shape = in_shape[0]
		label_shape = (in_shape[0][0],)
		output_shape = in_shape[0]
		return [data_shape, label_shape], [output_shape], []
	#  The infer_shape function should always return three lists in this order: inputs, outputs, and auxiliary states (which we don’t have here), even if one of them is empty.
	def infer_type(self, in_type):
   	 	dtype = in_type[0]
    	return [dtype, dtype], [dtype], []

	def create_operator(self, ctx, shapes, dtypes):
		return Softmax()

## Called method
mlp = mx.symbol.Custom(data=fc3, name='softmax', op_type='softmax')
```
	The forward function takes a list of input and a list of output NDArrays. For convenience, we called .asnumpy() on the first NDArray in input and convert it to a CPU-based NumPy array. This can be very slow. If you want the best performance, keep data in the NDArray format and use operators under mx.nd to do the computation.

## 预处理数据
基本的简单数据在`mxnet.test_utils`都可以找到，可以用来测试模型正确性。
`mx.io.DataIter`
## Fine-tune with Pretrained Models

### 多种

* 作为feature extractor https://github.com/dmlc/mxnet-notebooks/blob/master/python/how_to/predict.ipynb
* update all of the network’s weights for the new task

## 参考地址

[^1]: https://github.com/apache/incubator-mxnet "MXNet源代码下载"
[^2]: https://mxnet.incubator.apache.org/install/index.html "安装文档"
[^3]: http://www.jianshu.com/p/b9ce6258a75d " 基础结构解析 "
[^4]: http://www.cnblogs.com/McKean/p/6391380.html