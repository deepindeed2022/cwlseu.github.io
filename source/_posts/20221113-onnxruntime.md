---
layout: post
title: OnnxRuntime
categories: [blog]
tags: [Backends]
date: 2022-11-13 19:39:39
description: OnnxRuntime
---


## 一、ONNX简介

- 什么是ONNX？

  [ONNX](https://github.com/onnx/onnx)全称为 Open Neural Network Exchange，是一种与框架无关的模型表达式。ONNX的规范及代码主要由微软，亚马逊 ，Facebook 和 IBM 等公司共同开发，以开放源代码的方式托管在Github上。目前官方支持加载ONNX模型并进行推理的深度学习框架有： Caffe2, PyTorch, MXNet，ML.NET，TensorRT 和 Microsoft CNTK，并且 TensorFlow 也非官方的支持ONNX。

- ONNX的数据格式是怎么样的？

  ONNX本质上一种文件格式，通过Protobuf数据结构存储了神经网络结构权重。其组织格式核心定义在于[onnx.proto](https://github.com/onnx/onnx/blob/master/onnx/onnx.proto)，其中定义了Model/Graph/Node/ValueInfo/Tensor/Attribute层面的数据结构。整图通过各节点（Node）的input/output指向关系构建模型图的拓扑结构。

- ONNX支持的功能？

  基于ONNX模型，官方提供了一系列相关工具：模型转化/模型优化（[simplifier](https://github.com/daquexian/onnx-simplifier)等）/模型部署([Runtime](https://github.com/microsoft/onnxruntime))/模型可视化（[Netron](https://github.com/onnx/onnx/blob/master/onnx/onnx.proto)等）等

## 二、OnnxRuntime

ONNX自带了Runtime库，能够将ONNX Model部署到不同的硬件设备上进行推理，支持各种后端（如TensorRT/OpenVINO）。

基于ONNX Model的Runtime系统架构[^1]如下，可以看到Runtime实现功能是将ONNX Model转换为In-Memory Graph格式，之后通过将其转化为各个可执行的子图，最后通过`GetCapability()` API将子图分配到不同的后端（execution provider, EP）执行。

<img src="https://cdn.jsdelivr.net/gh/cwlseu/deepindeed_repo@main/images/onnxruntime_1.png" alt="onnxruntime_1" style="zoom:67%;" />

### 已支持的Execution Providers

| CPU                                                          | GPU                                                          | IoT/Edge/Mobile                                              | Other                                                        |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Default CPU                                                  | [NVIDIA CUDA](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html) | [Intel OpenVINO](https://onnxruntime.ai/docs/execution-providers/OpenVINO-ExecutionProvider.html) | [Rockchip NPU](https://onnxruntime.ai/docs/execution-providers/community-maintained/RKNPU-ExecutionProvider.html) (*preview*) |
| [Intel DNNL](https://onnxruntime.ai/docs/execution-providers/oneDNN-ExecutionProvider.html) | [NVIDIA TensorRT](https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html) | [ARM Compute Library](https://onnxruntime.ai/docs/execution-providers/community-maintained/ACL-ExecutionProvider.html) (*preview*) | [Xilinx Vitis-AI](https://onnxruntime.ai/docs/execution-providers/community-maintained/Vitis-AI-ExecutionProvider.html) (*preview*) |
| [TVM](https://onnxruntime.ai/docs/execution-providers/community-maintained/TVM-ExecutionProvider.html) (*preview*) | [DirectML](https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html) | [Android Neural Networks API](https://onnxruntime.ai/docs/execution-providers/NNAPI-ExecutionProvider.html) | [Huawei CANN](https://onnxruntime.ai/docs/execution-providers/community-maintained/CANN-ExecutionProvider.html) (*preview*) |
| [Intel OpenVINO](https://onnxruntime.ai/docs/execution-providers/OpenVINO-ExecutionProvider.html) | [AMD MIGraphX](https://onnxruntime.ai/docs/execution-providers/community-maintained/MIGraphX-ExecutionProvider.html) (*preview*) | [ARM-NN](https://onnxruntime.ai/docs/execution-providers/community-maintained/ArmNN-ExecutionProvider.html) (*preview*) |                                                              |
|                                                              | [AMD ROCm](https://onnxruntime.ai/docs/execution-providers/ROCm-ExecutionProvider.html) (*preview*) | [CoreML](https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html) (*preview*) |                                                              |
|                                                              | [TVM](https://onnxruntime.ai/docs/execution-providers/community-maintained/TVM-ExecutionProvider.html) (*preview*) | [TVM](https://onnxruntime.ai/docs/execution-providers/community-maintained/TVM-ExecutionProvider.html) (*preview*) |                                                              |
|                                                              | [Intel OpenVINO](https://onnxruntime.ai/docs/execution-providers/OpenVINO-ExecutionProvider.html) | [Qualcomm SNPE](https://onnxruntime.ai/docs/execution-providers/SNPE-ExecutionProvider.html) |                                                              |
| [XNNPACK](https://onnxruntime.ai/docs/execution-providers/Xnnpack-ExecutionProvider.html) |                                                              | [XNNPACK](https://onnxruntime.ai/docs/execution-providers/Xnnpack-ExecutionProvider.html) |                                                              |

我们最常用的是`CPUExecutionProvider`和`CUDAExecutionAProvider`, 如果要自己实现一个EP，可以参考[^2]





## Runtime源码剖析

### onnx中如何绑定C与python接口的

- onnx_runtime/onnxruntime/python/onnxruntime_pybind_state.h中声明了三个接口，这个三个接口实现C与python接口的绑定

```python
#include "onnxruntime_pybind.h"  // must use this for the include of <pybind11/pybind11.h>

namespace onnxruntime {
namespace python {

void addGlobalMethods(py::module& m, Environment& env);
void addObjectMethods(py::module& m, Environment& env);
void addOrtValueMethods(pybind11::module& m);

}  // namespace python
}  // namespace onnxruntime


// onnxruntime/onnxruntime/python/onnxruntime_pybind_module.cc中通过

#include "onnxruntime_pybind.h"  // must use this for the include of <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "core/providers/get_execution_providers.h"

namespace onnxruntime {
namespace python {
namespace py = pybind11;

void CreateInferencePybindStateModule(py::module& m);

PYBIND11_MODULE(onnxruntime_pybind11_state, m) {
  CreateInferencePybindStateModule(m);
  // move it out of shared method since training build has a little different behavior.
  m.def(
      "get_available_providers", []() -> const std::vector<std::string>& { return GetAvailableExecutionProviderNames(); },
      "Return list of available Execution Providers in this installed version of Onnxruntime. "
      "The order of elements represents the default priority order of Execution Providers "
      "from highest to lowest.");
}
}  // namespace python
}  // namespace onnxruntime
```

InferenceSession中调用构造函数，但是没有这个类型的构造函数

```C++
return onnxruntime::make_unique<InferenceSession>(so, arg, SessionObjectInitializer::Get());

  /**
    Create a new InferenceSession
    @param session_options Session options.
    @param model_uri absolute path of the model file.
    @param logging_manager
    Optional logging manager instance that will enable per session logger output using
    session_options.session_logid as the logger id in messages.
    If nullptr, the default LoggingManager MUST have been created previously as it will be used
    for logging. This will use the default logger id in messages.
    See core/common/logging/logging.h for details, and how LoggingManager::DefaultLogger works.
    This ctor will throw on encountering model parsing issues.
    */
  InferenceSession(const SessionOptions& session_options,
                   const std::string& model_uri,
                   logging::LoggingManager* logging_manager = nullptr);
```

`SessionObjectInitializer::Get()`返回一个`SessionObjectInitializer`类型的对象， 这个对象不得了，开发者给他订一了两个类型转换函数，

- `SessionObjectInitializer -> const SesssionOptions& Args1`

- `SessionObjectInitializer->logging::LoggingManager*`

这样编译器就可以利用这两个类型转换函数来实现类型的隐式转换而没有报类型错误。

```C++
class SessionObjectInitializer {
 public:
  typedef const SessionOptions& Arg1;
  typedef logging::LoggingManager* Arg2;
  operator Arg1() {
    return GetDefaultCPUSessionOptions();
  }

  operator Arg2() {
    static std::string default_logger_id{"Default"};
    static LoggingManager default_logging_manager{std::unique_ptr<ISink>{new CErrSink{}},
                                                  Severity::kWARNING, false, LoggingManager::InstanceType::Default,
                                                  &default_logger_id};
    return &default_logging_manager;
  }

  static SessionObjectInitializer Get() {
    return SessionObjectInitializer();
  }
};
```

终于找到真身了‼️

- model_path初始化
- model_proto初始化
- 构建session

```C++
InferenceSession::InferenceSession(const SessionOptions& session_options,
                                   const std::string& model_uri,
                                   logging::LoggingManager* logging_manager)
    : insert_cast_transformer_("CastFloat16Transformer") {
  model_location_ = ToWideString(model_uri);
  model_proto_ = onnxruntime::make_unique<ONNX_NAMESPACE::ModelProto>();
  auto status = Model::Load(model_location_, *model_proto_);
  ORT_ENFORCE(status.IsOK(), "Given model could not be parsed while creating inference session. Error message: ",
              status.ErrorMessage());

  // Finalize session options and initialize assets of this session instance
  ConstructorCommon(session_options, logging_manager);
}
```



### `Model::Load`加载模型的流程

- 不支持的算子如何处理？

- 多个provider之间怎么分配运行图

  

### 如何注册一个Execution Provider



## FAQ

#### 01、`clang: error: argument unused during compilation: '-mfpu=neon' [-Werror,-Wunused-command-line-argument]`

这个应该是由于onnxruntime对MAC M1机型的兼容性的一个bug，当前我先临时将`-mfpu=neon`从`CMAKE_CXX_FLAGS`中移除，然后重新编译。

```bash
[ 20%] Building CXX object CMakeFiles/onnxruntime_common.dir/Users/xxx/Documents/Framework/onnxruntime/onnxruntime/core/common/cpuid_info.cc.o
clang: error: argument unused during compilation: '-mfpu=neon' [-Werror,-Wunused-command-line-argument]
make[2]: *** [CMakeFiles/onnxruntime_common.dir/Users/xxx/Documents/Framework/onnxruntime/onnxruntime/core/common/cpuid_info.cc.o] Error 1
make[1]: *** [CMakeFiles/onnxruntime_common.dir/all] Error 2
make: *** [all] Error 2
Traceback (most recent call last):
  File "/Users/xxx/Documents/Framework/onnxruntime/tools/ci_build/build.py", line 1065, in <module>
    sys.exit(main())
  File "/Users/xxx/Documents/Framework/onnxruntime/tools/ci_build/build.py", line 1002, in main
    build_targets(args, cmake_path, build_dir, configs, args.parallel)
  File "/Users/xxx/Documents/Framework/onnxruntime/tools/ci_build/build.py", line 471, in build_targets
    run_subprocess(cmd_args, env=env)
  File "/Users/xxx/Documents/Framework/onnxruntime/tools/ci_build/build.py", line 212, in run_subprocess
    completed_process = subprocess.run(args, cwd=cwd, check=True, stdout=stdout, stderr=stderr, env=my_env, shell=shell)
  File "/Users/xxx/mambaforge/lib/python3.9/subprocess.py", line 528, in run
    raise CalledProcessError(retcode, process.args,
subprocess.CalledProcessError: Command '['/opt/homebrew/Cellar/cmake/3.23.2/bin/cmake', '--build', '/Users/xxx/Documents/Framework/onnxruntime/build/Linux/Debug', '--config', 'Debug']' returned non-zero exit status 2.
```

<img src="https://cdn.jsdelivr.net/gh/cwlseu/deepindeed_repo@main/images/build_onnxruntime_osx.png" alt="image-20220712114712012" style="zoom: 67%;" />

```bash
/usr/local/lib/python3.8/dist-packages/torch/cuda/__init__.py:146: UserWarning:
NVIDIA A100 80GB PCIe with CUDA capability sm_80 is not compatible with the current PyTorch installation.
The current PyTorch install supports CUDA capabilities sm_37 sm_50 sm_60 sm_70.
If you want to use the NVIDIA A100 80GB PCIe GPU with PyTorch, please check the instructions at https://pytorch.org/get-started/locally/

  warnings.warn(incompatible_device_warn.format(device_name, capability, " ".join(arch_list), device_name))
Traceback (most recent call last):
  File "tools/export_onnx_models.py", line 117, in <module>
    convert_onnx(args.model_name, args.model_path, args.batch_size, export_fp16=args.fp16, verbose=args.verbose)
  File "tools/export_onnx_models.py", line 69, in convert_onnx
    inputs = torch.rand(batch_size, 3, 224, 224, dtype=dtype, device=0)
RuntimeError: CUDA error: no kernel image is available for execution on the device
```

#### 02、load model is DataParallel format, your should notice:

- the op name with 'module.' which will result some operator failed, such as `load_state_dict` will throw miss match

```python
#
# load model weight and use another model weight update the network weight
# 
model = EfficientNet.from_pretrained('efficientnet-b4', num_classes=5)
model.set_swish(memory_efficient=False)
dataparallel_model = torch.load(model_path, map_location="cpu")
from collections import OrderedDict
new_state_dict = OrderedDict()
# method 1: use module state_dict to update weight
for k in dataparallel_model.module.state_dict():
    new_state_dict[k] = dataparallel_model.module.state_dict()[k]

# method 2: current dataparallel_model weight is module._xxxname 
for k in dataparallel_model.state_dict():
    new_state_dict[k[7:]] = dataparallel_model.state_dict()[k]

model.load_state_dict(new_state_dict)
model.cuda()
torch.onnx.export(model, inputs, output_fn, verbose=verbose)
```

####  03、 Some operator not supported by ONNX

```bash
WARNING: The shape inference of prim::PythonOp type is missing, so it may result in wrong shape inference for the exported graph. Please consider adding it in symbolic function.
Traceback (most recent call last):
  File "export_onnx_efficient_cls.py", line 79, in <module>
    convert_onnx("efficient_b4_big_5cls", args.model_path, args.batch_size)
  File "export_onnx_efficient_cls.py", line 55, in convert_onnx
    torch.onnx.export(model.module, inputs, output_fn, verbose=verbose)
  File "/home/xxxx/software/miniconda3/envs/inference/lib/python3.8/site-packages/torch/onnx/__init__.py", line 350, in export
    return utils.export(
  File "/home/xxxx/software/miniconda3/envs/inference/lib/python3.8/site-packages/torch/onnx/utils.py", line 163, in export
    _export(
  File "/home/xxxx/software/miniconda3/envs/inference/lib/python3.8/site-packages/torch/onnx/utils.py", line 1110, in _export
    ) = graph._export_onnx(  # type: ignore[attr-defined]
RuntimeError: ONNX export failed: Couldn't export Python operator SwishImplementation

```

#### 04、 获取onnx模型的输出

```python
# get onnx output
input_all = [node.name for node in onnx_model.graph.input]
input_initializer = [
    node.name for node in onnx_model.graph.initializer
]
net_feed_input = list(set(input_all) - set(input_initializer))
assert (len(net_feed_input) == 1)
```

#### 05、TypeError: Descriptors cannot not be created directly.

```sh
Traceback (most recent call last):
  File "export_onnx_models.py", line 4, in <module>
    import onnx
  File "/home/xxx/software/miniconda3/envs/inference/lib/python3.8/site-packages/onnx/__init__.py", line 6, in <module>
    from onnx.external_data_helper import load_external_data_for_model, write_external_data_tensors, convert_model_to_external_data
  File "/home/xxx/software/miniconda3/envs/inference/lib/python3.8/site-packages/onnx/external_data_helper.py", line 9, in <module>
    from .onnx_pb import TensorProto, ModelProto, AttributeProto, GraphProto
  File "/home/xxx/software/miniconda3/envs/inference/lib/python3.8/site-packages/onnx/onnx_pb.py", line 4, in <module>
    from .onnx_ml_pb2 import *  # noqa
  File "/home/xxx/software/miniconda3/envs/inference/lib/python3.8/site-packages/onnx/onnx_ml_pb2.py", line 33, in <module>
    _descriptor.EnumValueDescriptor(
  File "/home/xxx/software/miniconda3/envs/inference/lib/python3.8/site-packages/google/protobuf/descriptor.py", line 755, in __new__
    _message.Message._CheckCalledFromGeneratedFile()
TypeError: Descriptors cannot not be created directly.
If this call came from a _pb2.py file, your generated code is out of date and must be regenerated with protoc >= 3.19.0.
If you cannot immediately regenerate your protos, some other possible workarounds are:
 1. Downgrade the protobuf package to 3.20.x or lower.
 2. Set PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python (but this will use pure-Python parsing and will be much slower).

More information: https://developers.google.com/protocol-buffers/docs/news/2022-05-06#python-updates
```

protobuf版本太高与现有的onnxparser不兼容，根据错误提示降低protobuf的版本即可。
python3 -m pip install protobuf==3.19.4

#### 06、AttributeError: 'Upsample' object has no attribute 'recompute_scale_factor'

```sh
Traceback (most recent call last):
  File "export_onnx_models.py", line 148, in <module>
    convert_onnx(args.model_name, args.model_path, batch_size=args.batch_size, image_size=args.img_size, export_fp16=args.fp16, simplify=args.simplify, verify=args.verify, verbose=args.verbose)
  File "export_onnx_models.py", line 75, in convert_onnx
    test_infer_performance(model=model, model_name=model_name, batch_size=batch_size, input_shape=(3, image_size, image_size), num_data=10240)
  File "/home/xxx/Repo/infra_utilities/model_utils.py", line 72, in test_infer_performance
    ret = model(data)
  File "/home/xxx/software/miniconda3/envs/inference/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1110, in _call_impl
    return forward_call(*input, **kwargs)
  File "/home/xxx/Repo/infra_utilities/./models/yolox/models/yolox.py", line 30, in forward
    fpn_outs = self.backbone(x)
  File "/home/xxx/software/miniconda3/envs/inference/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1110, in _call_impl
    return forward_call(*input, **kwargs)
  File "/home/xxx/Repo/infra_utilities/./models/yolox/models/yolo_pafpn.py", line 98, in forward
    f_out0 = self.upsample(fpn_out0)  # 512/16
  File "/home/xxx/software/miniconda3/envs/inference/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1110, in _call_impl
    return forward_call(*input, **kwargs)
  File "/home/x x x/software/miniconda3/envs/inference/lib/python3.8/site-packages/torch/nn/modules/upsampling.py", line 154, in forward
    recompute_scale_factor=self.recompute_scale_factor)
  File "/home/xxx/software/miniconda3/envs/inference/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1185, in __getattr__
    raise AttributeError("'{}' object has no attribute '{}'".format(
AttributeError: 'Upsample' object has no attribute 'recompute_scale_factor'
```

torch版本降低到版本1.9.1，torchvision版本降低到版本0.10.1。但是我是通过在torch代码里进行更改进行解决。

https://github.com/pytorch/pytorch/pull/43535/files

#### 07、RuntimeError: Input type (torch.cuda.FloatTensor) and weight type (torch.cuda.HalfTensor) should be the same

转化数据格式类型即可

#### 08、ERROR - In node 23 (parseGraph): INVALID_NODE: Invalid Node - Pad_23

[shuffleNode.cpp::symbolicExecute::392] Error Code 4: Internal Error (Reshape_12: IShuffleLayer applied to shape tensor must have 0 or 1 reshape dimensions: dimensions were [-1,2])

因为tensorrt 截止到版本8.4.1为止，对动态输入的情况下，这种输入不支持导致的。遇到这种问题，当前只能先取消对动态输入的支持，采用固定shape的输入。

![image-20220909095831552](https://cdn.jsdelivr.net/gh/cwlseu/deepindeed_repo@main/images/image-20220909095831552.png)

![image-20220909095846591](../../../Library/Application Support/typora-user-images/image-20220909095846591.png)

python3 -m pip install torch==1.12.0+cu115 -f https://download.pytorch.org/whl/torch_stable.html

## 参考链接

[^1]: https://onnxruntime.ai/docs/execution-providers/ "onnxruntime execution"
[^2]: https://onnxruntime.ai/docs/execution-providers/add-execution-provider.html "add execution provider"

