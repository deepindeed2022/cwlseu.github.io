安装新的操作系统，配置开发环境需要安装如下开发依赖包

## install vim
sudo apt-get install vim  git

## tensorflow virtualenv
https://www.tensorflow.org/install/install_linux#InstallingVirtualenv

```sh
sudo apt-get install python-pip python-dev python-virtualenv
virtualenv --system-site-packages tensorflow
source ~/tensorflow/bin/activate
(tensorflow)$ pip install --upgrade tensorflow
```

## ML package

```sh
wget https://repo.continuum.io/archive/Anaconda2-4.4.0-Linux-x86_64.sh
chmod +x Anaconda2-4.4.0-Linux-x86_64.sh
./Anaconda2-4.4.0-Linux-x86_64.sh
## 根据提示填写内容

## 添加内容到.bashrc
export PATH=/opt/anaconda2/bin:$PATH
export PYTHONPATH=/opt/anaconda2/lib/python2.7/site-packages:$PYTHONPATH

```

## opencv