---
layout: post
title: "玩转Caffe之Dockerfile"
categories: [blog ]
tags: [Caffe, ]
description:  租用亚马逊的环境进行实验，直接运行docker file 进行镜像创建
---
声明：本博客欢迎转发，但请保留原作者信息!                                      
作者: [曹文龙]                                                                 
博客： <https://cwlseu.github.io/>                                             

[TOC]

## Caffe standalone Dockerfiles.

The `standalone` subfolder contains docker files for generating both CPU and GPU executable images for Caffe. The images can be built using make, or by running:

```
docker build -t caffe:cpu standalone/cpu
```
for example. (Here `gpu` can be substituted for `cpu`, but to keep the readme simple, only the `cpu` case will be discussed in detail).

Note that the GPU standalone requires a CUDA 7.5 capable driver to be installed on the system and [nvidia-docker] for running the Docker containers. Here it is generally sufficient to use `nvidia-docker` instead of `docker` in any of the commands mentioned.

# Running Caffe using the docker image

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

## Other use-cases

Although running the `caffe` command in the docker containers as described above serves many purposes, the container can also be used for more interactive use cases. For example, specifying `bash` as the command instead of `caffe` yields a shell that can be used for interactive tasks. (Since the caffe build requirements are included in the container, this can also be used to build and run local versions of caffe).

Another use case is to run python scripts that depend on `caffe`'s Python modules. Using the `python` command instead of `bash` or `caffe` will allow this, and an interactive interpreter can be started by running:
```
docker run -ti caffe:cpu python
```
(`ipython` is also available in the container).

Since the `caffe/python` folder is also added to the path, the utility executable scripts defined there can also be used as executables. This includes `draw_net.py`, `classify.py`, and `detect.py`

## 挂载本地目录到容器中
```bash
cd cafferoot

docker -run -ti --volume=$(pwd):/workspace caffe:cpu /bin/bash

```

解析：`--volume=$(pwd):/workspace`是挂载本机目录到容器中，`--volume or -v`是docker的挂载命令，`=$(pwd):/workspace`是挂载信息，是将`$(pwd)`即本机当前目录，:是挂载到哪，`/workspace`是容器中的目录，就是把容器中的`workspace`目录换成本机的当前目录，这样就可以在本机与容器之间进行交互了，本机当前目录可以编辑，容器中同时能看到。容器中的workspace目录的修改也直接反应到了本机上。`$()`是Linux中的命令替换，即将$()中的命令内容替换为参数，pwd是Linux查看当前目录，我的本机当前目录为cafferoot，`--volume=$(pwd):/workspace`就等于`--volume=/Users/***/cafferoot:/workspace`，`/Users/***/cafferoot`为`pwd`的执行结果，$()是将pwd的执行结果作为参数执行。

## reference

1. [Docker理论与实践]<http://noahsnail.com/2016/12/01/2016-12-1-Docker%E7%90%86%E8%AE%BA%E4%B8%8E%E5%AE%9E%E8%B7%B5%EF%BC%88%E5%9B%9B%EF%BC%89/>
2. [docker install caffe]<https://github.com/cwlseu/caffe/edit/master/docker/README.md>