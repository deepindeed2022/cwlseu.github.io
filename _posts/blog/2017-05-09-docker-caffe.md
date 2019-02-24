---
layout: post
title: "深度学习：玩转Caffe之Dockerfile"
categories: [blog ]
tags: [Caffe, 深度学习]
description:  租用亚马逊的环境进行实验，直接运行docker file 进行镜像创建
---
{:toc}

- 声明：本博客欢迎转发，但请保留原作者信息!
- 作者: [曹文龙]
- 博客： <https://cwlseu.github.io/>

## Docker安装caffe

The `standalone` subfolder contains docker files for generating both CPU and GPU executable images for Caffe. The images can be built using make, or by running: `docker build -t caffe:cpu standalone/cpu`
for example. (Here `gpu` can be substituted for `cpu`, but to keep the readme simple, only the `cpu` case will be discussed in detail).

Note that the GPU standalone requires a CUDA 8.0 capable driver to be installed on the system and [nvidia-docker] for running the Docker containers. Here it is generally sufficient to use `nvidia-docker` instead of `docker` in any of the commands mentioned.

```Dockerfile
# 基础镜像
FROM ubuntu:14.04
# 进行维护者信息
MAINTAINER caffe-maint@googlegroups.com
# 在基础镜像上执行一些命令，安装caffe依赖的libs
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        wget \
        libatlas-base-dev \
        libboost-all-dev \
        libgflags-dev \
        libgoogle-glog-dev \
        libhdf5-serial-dev \
        libleveldb-dev \
        liblmdb-dev \
        libopencv-dev \
        libprotobuf-dev \
        libsnappy-dev \
        protobuf-compiler \
        python-dev \
        python-numpy \
        python-pip \
        python-scipy && \
    rm -rf /var/lib/apt/lists/*

# 声明变量并初始化
ENV CAFFE_ROOT=/opt/caffe
# 切换当前工作路径为
WORKDIR $CAFFE_ROOT

# FIXME: clone a specific git tag and use ARG instead of ENV once DockerHub supports this.
ENV CLONE_TAG=master
# clone caffe源代码，安装python依赖库，编译caffe
RUN git clone -b ${CLONE_TAG} --depth 1 https://github.com/BVLC/caffe.git . && \
    for req in $(cat python/requirements.txt) pydot; do pip install $req; done && \
    mkdir build && cd build && \
    cmake -DCPU_ONLY=1 .. && \
    make -j"$(nproc)"

# 设置环境变量
ENV PYCAFFE_ROOT $CAFFE_ROOT/python
ENV PYTHONPATH $PYCAFFE_ROOT:$PYTHONPATH
ENV PATH $CAFFE_ROOT/build/tools:$PYCAFFE_ROOT:$PATH
# 像链接文件中写入当前caffe库
RUN echo "$CAFFE_ROOT/build/lib" >> /etc/ld.so.conf.d/caffe.conf && ldconfig

# 切换工作路径
WORKDIR /workspace
```

## 在docker镜像中运行caffe

In order to test the Caffe image, run:
`docker run -ti caffe:cpu caffe --version`

which should show a message like:

```sh
libdc1394 error: Failed to initialize libdc1394
caffe version 1.0.0-rc3
```

One can also build and run the Caffe tests in the image using:

`docker run -ti caffe:cpu bash -c "cd /opt/caffe/build; make runtest"` 

In order to get the most out of the caffe image, some more advanced `docker run` options could be used. For example, running:

`docker run -ti --volume=$(pwd):/workspace caffe:cpu caffe train --solver=example_solver.prototxt`

will train a network defined in the `example_solver.prototxt` file in the current directory (`$(pwd)` is maped to the container volume `/workspace` using the `--volume=` Docker flag).

Note that docker runs all commands as root by default, and thus any output files (e.g. snapshots) generated will be owned by the root user. In order to ensure that the current user is used instead, the following command can be used:

`docker run -ti --volume=$(pwd):/workspace -u $(id -u):$(id -g) caffe:cpu caffe train --solver=example_solver.prototxt`

where the `-u` Docker command line option runs the commands in the container as the specified user, and the shell command `id` is used to determine the user and group ID of the current user. Note that the Caffe docker images have `/workspace` defined as the default working directory. This can be overridden using the `--workdir=` Docker command line option.

## 其他案例

Although running the `caffe` command in the docker containers as described above serves many purposes, the container can also be used for more interactive use cases. For example, specifying `bash` as the command instead of `caffe` yields a shell that can be used for interactive tasks. (Since the caffe build requirements are included in the container, this can also be used to build and run local versions of caffe).

Another use case is to run python scripts that depend on `caffe`'s Python modules. Using the `python` command instead of `bash` or `caffe` will allow this, and an interactive interpreter can be started by running:

`docker run -ti caffe:cpu python`

(`ipython` is also available in the container).

Since the `caffe/python` folder is also added to the path, the utility executable scripts defined there can also be used as executables. This includes `draw_net.py`, `classify.py`, and `detect.py`

## 挂载本地目录到容器中

```bash
cd cafferoot

docker -run -ti --volume=$(pwd):/workspace caffe:cpu /bin/bash

```

解析：`--volume=$(pwd):/workspace`是挂载本机目录到容器中，`--volume or -v`是docker的挂载命令，`=$(pwd):/workspace`是挂载信息，是将`$(pwd)`即本机当前目录，:是挂载到哪，`/workspace`是容器中的目录，就是把容器中的`workspace`目录换成本机的当前目录，这样就可以在本机与容器之间进行交互了，本机当前目录可以编辑，容器中同时能看到。容器中的workspace目录的修改也直接反应到了本机上。`$()`是Linux中的命令替换，即将$()中的命令内容替换为参数，pwd是Linux查看当前目录，我的本机当前目录为cafferoot，`--volume=$(pwd):/workspace`就等于`--volume=/Users/***/cafferoot:/workspace`，`/Users/***/cafferoot`为`pwd`的执行结果，$()是将pwd的执行结果作为参数执行。

## 一些有用的命令

1. 基于image `nvidia/cuda`运行某条命令`nvidia-smi`
`nvidia-docker run  nvidia/cuda nvidia-smi`
2. 查看当前有哪些镜像
`sudo nvidia-docker images`
3. 查看当前有哪些运行中的实例
`sudo nvidia-docker ps -l`
4. 运行某个镜像的实例
`sudo nvidia-docker run -it nvidia/cuda`
5. 链接运行中的镜像
`sudo nvidia-docker attach d641ab33bec2`
6. 跳出运行中的image镜像，但是不退出
`ctrl+p`, `ctrl+q`

7. 使用linux命令对镜像实例进行操作
`sudo nvidia-docker cp zeyu/VOCtrainval_11-May-2012.tar  e6a0961ab4cf:/workspace/data`
`sudo nvidia-docker run -it -t nvidia/cuda nvidia-smi`

8. 在host机器上启动新的bash
`sudo nvidia-docker exec -it d641ab33bec2 bash`



## TLS handshake timeout 问题

iscas@ZXC-Lenovo:~/Repo$ sudo docker build -t caffe:cpu docker/caffe
Sending build context to Docker daemon 3.072 kB
Step 1/12 : FROM ubuntu:14.04
Get https://registry-1.docker.io/v2/: net/http: TLS handshake timeout

很明显可以看出是连接不到 docker hub，那就需要查看网络原因了。当然较简单的解决办法就是用国内的仓库，
下面的方法就是使用国内的 daocloud 的仓库：

> 添加国内库的依赖

`$ echo "DOCKER_OPTS="$DOCKER_OPTS --registry-mirror=http://f2d6cb40.m.daocloud.io"" | sudo tee -a /etc/default/docker`

> 重启服务

`$ sudo service docker restart`

更多问题可以查看[3]

## reference

1. [Docker理论与实践]<http://noahsnail.com/2016/12/01/2016-12-1-Docker%E7%90%86%E8%AE%BA%E4%B8%8E%E5%AE%9E%E8%B7%B5%EF%BC%88%E5%9B%9B%EF%BC%89/>
2. [docker install caffe]<https://github.com/cwlseu/caffe/edit/master/docker/README.md>

3. [Pull Docker image的时候遇到docker pull TLS handshake timeout]<https://blog.csdn.net/han_cui/article/details/55190319>

4. [caffe cpu docker]<https://hub.docker.com/r/elezar/caffe/>