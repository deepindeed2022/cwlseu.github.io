---
layout: post
title: Elasticsearch
categories: [blog,]
tags: [Elasticsearch, Docker]
description: 
---

## Start
1. install java
2. install pluge

*Qurst*. Couldn't find any executable java binary.
```
    charles@ubuntu:/opt/elasticsearch-2.1.1$ sudo ./bin/plugin -i elasticsearch/marvel/lastest
    Could not find any executable java binary. Please install java in your PATH or set JAVA_HOME
```
*Solution*

    bin/plugin install license
    bin/plugin install marvel-agent
一些类似的问题的解决方案，请参考[Install Marvel into Elasticsearch](http://stackoverflow.com/questions/23604868/install-marvel-plugin-for-elasticsearch)

