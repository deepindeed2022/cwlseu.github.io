---
layout: post
title: Effective Python
categories: [blog ]
tags: [Python, ]
description:  python中的语法糖
---

# Python 中不为人知的一面

[TOC]

1.  `eval` 与`ast.literal_eval`:
       literal_eval相对来说比较安全，只有字符串中包含表达式的时候才会评估。
2. 实现一组字符串分组分开：`itertools.groupby` 
```
from itertools import groupby
s = "1123433364433"
print([''.join(i) for _, i in groupby(s)])
```

3. 判断list是否为空
```
if not l:
	pass
```
就是最好的选择
[how-do-i-check-if-a-list-is-empty](https://stackoverflow.com/questions/53513/how-do-i-check-if-a-list-is-empty?rq=1)

4. 从json文件中读取参数
[parse values from a json file](https://stackoverflow.com/questions/2835559/parsing-values-from-a-json-file?noredirect=1&lq=1)

5. 各种`@property`的实现
[python property的作用](https://stackoverflow.com/questions/17330160/how-does-the-property-decorator-work)
[python property的重定义](https://stackoverflow.com/questions/3012421/python-memoising-deferred-lookup-property-decorator)

