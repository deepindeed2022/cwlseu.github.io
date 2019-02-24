---
layout: post
title: "Python开发中的问题（一）"
categories: [blog ]
tags: [Python]
description: Python开发中的问题记录。
---
{:toc}

- 声明：本博客欢迎转发，但请保留原作者信息!
- 作者: [曹文龙]
- 博客： <https://cwlseu.github.io/>

## pip安装package出现Read timed out.

```sh
xxx@xxxx:~/Repo/engine/online_index/webpy-master$ pip install cheroot-6.0.0-py2.py3-none-any.whl 
Processing ./cheroot-6.0.0-py2.py3-none-any.whl
Requirement already satisfied: six>=1.11.0 in /opt/anaconda2/lib/python2.7/site-packages (from cheroot==6.0.0)
Collecting more-itertools>=2.6 (from cheroot==6.0.0)
Retrying (Retry(total=4, connect=None, read=None, redirect=None)) after connection broken by 'ReadTimeoutError("HTTPSConnectionPool(host='pypi.python.org', port=443): Read timed out. (read timeout=15)",)': /simple/more-itertools/
Retrying (Retry(total=3, connect=None, read=None, redirect=None)) after connection broken by 'ReadTimeoutError("HTTPSConnectionPool(host='pypi.python.org', port=443): Read timed out. (read timeout=15)",)': /simple/more-itertools/
Retrying (Retry(total=2, connect=None, read=None, redirect=None)) after connection broken by 'ReadTimeoutError("HTTPSConnectionPool(host='pypi.python.org', port=443): Read timed out. (read timeout=15)",)': /simple/more-itertools/
Retrying (Retry(total=1, connect=None, read=None, redirect=None)) after connection broken by 'ReadTimeoutError("HTTPSConnectionPool(host='pypi.python.org', port=443): Read timed out. (read timeout=15)",)': /simple/more-itertools/

```

### 解决方案

安装过程中添加信赖的地址，尤其是在某些互联网公司中，由于安全，防火墙等等安全考虑，会将pip默认的host地址作为不信任。
xxx@xxxx:~/Repo/engine/online_index/webpy-master$ pip install web.py -i http://pypi.douban.com/simple --trusted-host pypi.douban.com

在pip.conf中加入trusted-host选项，该方法是一劳永逸
```git-config
[global]
index-url = http://mirrors.aliyun.com/pypi/simple/
[install]
trusted-host=mirrors.aliyun.com
```