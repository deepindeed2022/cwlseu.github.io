---
layout: post
title: "Linux开发中的问题录"
categories: [blog ]
tags: [Linux开发]
description: 
---


## fatal error: metis.h: No such file or directory
### NOT sudo user
I am trying to install Metis. Since I am working on a public server, I couldn't install it as a root user. So I have installed metis in my account /home/jd/metis.

When I try to execute something, I get

> fatal error: metis.h: No such file or directory

I guess the system looks for metis.h under /usr/local/include but couldnt find it there. How do I make linux look for metis.h under /home/jd/metis/include directory?
I added this path to the $PATH variable. But still the same error. Please advise.

Work with cmake. Adding `include_directories("/home/xxx/metis/include")`

### sudo user
参看[stack-overflow](http://stackoverflow.com/questions/36046189/how-to-install-metis-on-ubuntu/41336362#41336362)