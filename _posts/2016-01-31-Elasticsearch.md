---
layout: post
title: Elasticsearch
categories: [blog,]
tags: [Elasticsearch]
description: 
---


## install java
1. download java source from oracle
2. unpacking the source code
3. add environment value to /etc/profile 

    export JAVA_HOME=/opt/jdk1.8.0_71
    export JRE_HOME=/opt/jdk1.8.0_71/jre
    export CLASSPATH=.:$CLASSPATH:$JAVA_HOME/lib:$JRE_HOME/lib
    export PATH=$PATH:$JAVA_HOME/bin:$JRE_HOME/bin

4. checking the java system environment 
    java -version

*Qurst*. Couldn't find any executable java binary.
```
    charles@ubuntu:/opt/elasticsearch-2.1.1$ sudo ./bin/plugin -i elasticsearch/marvel/lastest
    Could not find any executable java binary. Please install java in your PATH or set JAVA_HOME
```
*Solution*

    bin/plugin install license
    bin/plugin install marvel-agent
一些类似的问题的解决方案，请参考[Install Marvel into Elasticsearch](http://stackoverflow.com/questions/23604868/install-marvel-plugin-for-elasticsearch)

## install docker
I have download a book called "Docker Cooking Book"