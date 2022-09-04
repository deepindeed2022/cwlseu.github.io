---
title: 工具
date: 2022-08-28 22:20:50
categories: "tools"
---

## Document
- [Pytorch](https://pytorch.org)

- [CppReference](https://cplusplus.com/reference)
- [NVIDIA CUDA](https://docs.nvidia.com/cuda/index.html)
- [gcc](https://gcc.gnu.org/onlinedocs/gccint/)
- [onnxruntime](https://onnxruntime.ai/about.html)

## 部署

- [nvidia-docker-container](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/user-guide.html)
- [ngc-pytorch](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch/tags)
- [pytorch modelzoo](https://pytorch.org/hub/)
- [timm modelzoo](https://rwightman.github.io/pytorch-image-models/)

## 可视化

- [netron](https://lutzroeder.github.io/netron/)
- [神经网络示意图绘制](https://www.zhihu.com/question/40698990?sort=created)
- [大麦地: 绘制思维导图、uml图](https://www.processon.com/diagrams)
- [diagrams：专门用来绘制软件工程相关插件软件的，支持google cloud/macro driver直接存储](https://app.diagrams.net/)

## 测试工具

> nsys + nsight测试GPU kernel的性能

```bash
export PATH=/opt/nvidia/nsight-systems/2020.4.1/bin:$PATH
/opt/nvidia/nsight-systems/2020.4.1/bin/nsys profile --stats=true -t cuda python xx.py
```

> 服务压测工具

`locust -f locust_file.py --host="" --headless -u 12 -r 20 -t 10m`

其中locust_file.py需要如下定义：
```python
import json
import glob
import random
import base64
from locust import HttpUser, task

fnames = list(glob.glob("./images/*"))
def build_request():
    fname = random.choice(fnames)
    buffer = open(fname, "rb").read()
    req = json.dumps({
        "image": base64.b64encode(buffer).decode("utf-8"),
    })
    return req


class MyUser(HttpUser):
    @task
    def process(self):
        req_data = build_request()
        with self.client.post("/api_name", data=req_data) as res:
            if res.status_code != 200:
                print("Didn't get response, got: " + str(res.status_code))

```

> 自动负载均衡
gunicorn 

