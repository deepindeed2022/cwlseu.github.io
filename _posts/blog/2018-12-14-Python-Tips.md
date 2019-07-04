---
layout: post
title: "Python中的一些奇淫技巧"
categories: [blog ]
tags: [Python, ]
description: Python入门简单，但是无论哪种语言，都有语言设计者赋予其中的特殊意义的东西，可能是为了方便，可能是为了得到高的level,无论出发点是什么，都是值得我学习探究的。
---
* content
{:toc}


## 前言

最近觉得 Python 太“简单了”，于是在师父川爷面前放肆了一把：“我觉得 Python 是世界上最简单的语言！”。于是川爷嘴角闪过了一丝轻蔑的微笑（内心 OS：Naive！，作为一个 Python 开发者，我必须要给你一点人生经验，不然你不知道天高地厚！）于是川爷给我了一份满分100分的题，然后这篇文章就是记录下做这套题所踩过的坑。

## 变量类型
这是关于list的数据类型的理解。觉得如下程序会输出怎么样的结果？

```python
def f(x, L = []):
	for i in range(x):
		L.append(i*i)
	print(L)
	print(id(L))

f(2)
f(2, [1,2,3])
f(3)
```
输出结果如下
```
[0, 1]
139664725541736
[1, 2, 3, 0, 1]
139664725491584
[0, 1, 0, 1, 4]
139664725541736
```

* 在python中的数据可以分为可变数据类型和不变数据类型。
可变数据类型：像tuple,list,dict之类的变量就是可变数据类型，变量名存储的是一个地址，
该地址指向一个具体的对象，并且不管对变量的值即对象做怎么样的操作，都不会改变变量名存储的地址。
我们把列表作为参数传入一个函数时，在函数内我们对该列表进行了一些改变，由于变量存储的地址没有变
，所以就算我们没有故意通过return语句把该列表传递出来，该列表还是会在函数执行结束后跟着改变的。

* 函数的默认值为可变数据类型的话，会申请一个变量，这个变量地址是固定的，但是值在函数可被改变，
而且还会影响下次调用。

同样的道理, 如果对不可变数据类型作为输入参数的话，会有什么结果？

```python
def g(x, s=""):
	for i in range(x):
		s += str("{} ".format(i))
	print(s)
	print(id(x), id(s))
g(2)
ss = "hello"
g(3, ss)
g(2)
```

输出结果如下：
```
0 1 
(42414400, 140100353397648)
140100353385936
hello0 1 2 
(42414376, 140100353397648)
0 1 
(42414400, 140100353397648)
```
从中可以看出，不可变数据类型的情况下，其中的
* 默认值是空没有随着函数调用被改变,与函数是绑定的关系，不信你可以测试一下相对位置是不变的；
* 而且在python语言的世界里，数字类型也是一个引用哦；

## 列表生成器

> 描述:

    下面的代码会报错，为什么？
```python
classA(object):
    x = 1
    gen = (xfor_inxrange(10))# gen=(x for _ in range(10))
    if__name__ == "__main__":
print(list(A.gen))
```

> 答案

这个问题是变量作用域问题，在 gen=(x for _ in xrange(10)) 中 gen 是一个 generator ,在 generator中变量有自己的一套作用域，与其余作用域空间相互隔离。因此，将会出现这样的 NameError: name 'x' is not defined 的问题，那么解决方案是什么呢？答案是：用 lambda 。

```python
classA(object):
    x = 1
    gen = (lambdax: (xfor_inxrange(10)))(x)# gen=(x for _ in range(10))
    if__name__ == "__main__":
print(list(A.gen))
```

## 装饰器

> 描述

我想写一个类装饰器用来度量函数/方法运行时间


```python
import time
class Timeit(object):
    def __init__(self, func):
        self._wrapped = func

    def __call__(self, *args, **kws):
        start_time = time.time()
        result = self._wrapped(*args, **kws)
        print("elapsed time is %s " % (time.time() - start_time))
        return result
```
这个装饰器能够运行在普通函数上：

```python
@Timeit
def func():
    time.sleep(1)
    return"invoking function func"

if__name__ == '__main__':
    func()# output: elapsed time is 1.00044410133
```

但是运行在方法上会报错，为什么？

```python
@Timeit
def func():
    time.sleep(1)
    return"invoking function func"

class A(object):
    @Timeit
    def func(self):
        time.sleep(1)
        return'invoking method func'

if __name__ == '__main__':

    a = A()
    a.func()# Boom!
```

如果我坚持使用类装饰器，应该如何修改？

> 答案

使用类装饰器后，在调用 `func` 函数的过程中其对应的 instance 并不会传递给 `__call__`方法，造成其 mehtod unbound ,那么解决方法是什么呢？描述符

```python
class Timeit(object):
    def __init__(self,func):
        self.func = func

    def __call__(self, *args, **kwargs):
        print('invoking Timer')

    def __get__(self, instance, owner):
        return lambda *args, **kwargs: self.func(instance, *args, **kwargs)

@Timeit
def func():
    time.sleep(1)
    return"invoking function func"

class A(object):
    @Timeit
    def func(self):
        time.sleep(1)
        return'invoking method func'

if __name__ == '__main__':
    a = A()
    print a.func()
```

## Python 调用机制

> 描述

我们知道 `__call__`方法可以用来重载圆括号调用，好的，以为问题就这么简单？Naive！

```python
class A(object):
    def __call__(self):
        print("invoking __call__ from A!")

if __name__ == "__main__":
    a = A()
    a()## output: invoking __call__ from A
```

现在我们可以看到 a() 似乎等价于 `a.__call__()`  ,看起来很 Easy 对吧，好的，我现在想作死，又写出了如下的代码，

`a.__call__ = lambda: "invoking __call__ from lambda"`
`a.__call__()`

\# output:invoking __call__ from lambda

`a()`

\# output:invoking __call__ from A!

请大佬们解释下，为什么 a() 没有调用出 `a.__call__()` (此题由 USTC 王子博前辈提出)
> 答案

原因在于，在 Python 中，新式类（ new class )的内建特殊方法，和实例的属性字典是相互隔离的，具体可以看看 Python 官方文档对于这一情况的说明

    For new-style classes, implicit invocations of special methods are only guaranteed to work correctly if defined on an object’s type, not in the object’s instance dictionary. That behaviour is the reason why the following code raises an exception (unlike the equivalent example with old-style classes):

同时官方也给出了一个例子：

```python
class C(object):
    pass
    # def __len__(self):
    #   return 5

c = C()
c.__len__ = lambda: 5
print len(c)
```

```
# Traceback (most recent call last):
#  File "<stdin>", line 1, in <module>
# TypeError: object of type 'C' has no len()
```

回到我们的例子上来，当我们在执行 `a.__call__=lambda: "invoking __call__ from lambda"`  时，的确在我们在 `a.__dict__` 中新增加了一个 key 为 `__call__` 的 item，但是当我们执行 a() 时，因为涉及特殊方法的调用，因此我们的调用过程不会从 `a.__dict__` 中寻找属性，而是从 `type(a).__dict__`中寻找属性。因此，就会出现如上所述的情况。

## 描述符

> 描述

我想写一个 Exam 类，其属性 math 为 [0,100] 的整数，若赋值时不在此范围内则抛出异常，我决定用描述符来实现这个需求。

```python
class Grade(object):
    def __init__(self):
        self._score = 0

    def __get__(self, instance, owner):
        return self._score

    def __set__(self, instance, value):
        if 0 <= value <= 100:
            self._score = value
        else:
            raise ValueError('grade must be between 0 and 100')

class Exam(object):
    math = Grade()

    def __init__(self, math):
        self.math = math

if __name__ == '__main__':
    niche = Exam(math = 90)
    print(niche.math)
    # output : 90
    snake = Exam(math = 75)
    print(snake.math)
    # output : 75
    snake.math = 120
    # output: ValueError:grade must be between 0 and 100!

```

看起来一切正常。不过这里面有个巨大的问题，尝试说明是什么问题。为了解决这个问题，我改写了 Grade 描述符如下：

```python
class Grade(object):
    def __init__(self):
        self._grade_pool = {}

    def __get__(self, instance, owner):
        return self._grade_pool.get(instance, None)

    def __set__(self, instance, value):
        if 0 <= value <= 100:
            _grade_pool = self.__dict__.setdefault('_grade_pool',{})
            _grade_pool[instance] = value
        else:
            raise ValueError("Ooh, Value Error")
```

不过这样会导致更大的问题，请问该怎么解决这个问题？

> 答案

1. 第一个问题的其实很简单，如果你再运行一次 print(niche.math) 你就会发现，输出值是 75 ，那么这是为什么呢？这就要先从 Python的调用机制说起了。我们如果调用一个属性，那么其顺序是优先从实例的 `__dict__` 里查找，然后如果没有查找到的话，那么依次查询类字典，父类字典，直到彻底查不到为止。好的，现在回到我们的问题，我们发现，在我们的类 Exam 中，其`self.math`的调用过程是，首先在实例化后的实例的`__dict__`中进行查找，没有找到，接着往上一级，在我们的类 Exam 中进行查找，好的找到了，返回。那么这意味着，我们对于 self.math 的所有操作都是对于类变量 math的操作。因此造成变量污染的问题。那么该则怎么解决呢？很多同志可能会说，恩，在 `__set__` 函数中将值设置到具体的实例字典不就行了。

那么这样可不可以呢？答案是，很明显不得行啊，至于为什么，就涉及到我们 Python 描述符的机制了，

> 描述符指的是实现了描述符协议的特殊的类，三个描述符协议指的是 `__get__ `, `__set__` , `__delete__`以及 Python 3.6 中新增的 `__set_name__` 方法，其中实现了`__get__` 以及 `__set__ / __delete__ / __set_name__` 的是 Data descriptors ，而只实现了 `__get__` 的是 Non-Data descriptor

那么有什么区别呢，前面说了， 我们如果调用一个属性，那么其顺序是优先从实例的 `__dict__` 里查找，然后如果没有查找到的话，那么一次查询类字典，父类字典，直到彻底查不到为止。 但是，这里没有考虑描述符的因素进去，如果将描述符因素考虑进去，那么正确的表述应该是我们如果调用一个属性，那么其顺序是优先从实例的 `__dict__` 里查找，然后如果没有查找到的话，那么依次查询类字典，父类字典，直到彻底查不到为止。其中如果在*类实例字典*中的该属性是一个 Data descriptors ，那么无论实例字典中存在该属性与否，无条件走描述符协议进行调用，在类实例字典中的该属性是一个Non-Data descriptors ，那么优先调用实例字典中的属性值而不触发描述符协议，如果实例字典中不存在该属性值，那么触发 Non-Data descriptor的描述符协议。回到之前的问题，我们即使在 `__set__`将具体的属性写入实例字典中，但是由于类字典中存在着 Data descriptors ，因此，我们在调用 math 属性时，依旧会触发描述符协议。

2. 经过改良的做法，利用dict的key唯一性，将具体的值与实例进行绑定，但是同时带来了内存泄露的问题。那么为什么会造成内存泄露呢，首先复习下我们的 dict 的特性:
* dict 最重要的一个特性，凡可 hash 的对象皆可为 key ，dict 通过利用的 hash 值的唯一性(严格意义上来讲并不是唯一，而是其 hash 值碰撞几率极小，近似认定其唯一)来保证 key 的不重复；
* 同时, dict中的key引用是*强引用类型*,会造成对应对象的引用计数的增加，可能造成对象无法被GC,从而产生内存泄露。

3. 那么这里该怎么解决呢？两种方法

* 第一种：

```python
class Grade(object):
    def __init__(self):
        import weakref
        self._grade_pool = weakref.WeakKeyDictionary()

    def __get__(self,instance,owner):
        return self._grade_pool.get(instance,None)

    def __set__(self,instance,value):
        if 0 <= value <= 100:
            _grade_pool = self.__dict__.setdefault('_grade_pool',{})
            _grade_pool[instance] = value
        else:

            raise ValueError("fuck")
class Exam(object):
    math = Grade()

    def __init__(self, math):
        self.math = math

if __name__ == '__main__':
    niche = Exam(math = 90)
    print(niche.math)
    # output : 90
    snake = Exam(math = 75)
    print(snake.math)
    # output : 75
    try:
        snake.math = 120
    except ValueError as e:
        print e
    print niche.math
```

weakref 库中的 WeakKeyDictionary 所产生的字典的 key 对于*对象的引用是弱引用类型*，其不会造成内存引用计数的增加，因此不会造成内存泄露。同理，如果我们为了避免 value 对于对象的强引用，我们可以使用 WeakValueDictionary 。

* 第二种：在**Python 3.6 **中，实现的PEP 487 提案，为描述符新增加了一个协议`__set_name__`，我们可以用其来绑定对应的对象：

```python
class Grade(object):
    def __get__(self, instance, owner):
        return instance.__dict__[self.key]

    def __set__(self, instance, value):
        if 0 <= value <= 100:
            instance.__dict__[self.key] = value
        else:
            raise ValueError("fuck")
    def __set_name__(self, owner, name):
        self.key = name

```

这道题涉及的东西比较多，这里给出一点参考链接
- invoking-descriptors(https://docs.python.org/2/reference/datamodel.html#invoking-descriptors)  
- Descriptor HowTo Guide(https://docs.python.org/3/howto/descriptor.html)  
- PEP 487(https://www.python.org/dev/peps/pep-0487/#adding-a-class-attribute-with-the-attribute-order) 
- what`s new in Python 3.6(https://docs.python.org/3.6/whatsnew/3.6.html) 

## Python 继承机制

> 描述

试求出以下代码的输出结果。

```python
class Init(object):
    def __init__(self, value):
        self.val = value
        print "init", self.val

class Add2(Init):
    def __init__(self, val):
        super(Add2, self).__init__(val)
        print "Add2 Before:", self.val
        self.val += 2
        print "Add2", self.val

class Mul5(Init):
    def __init__(self, val):
        super(Mul5, self).__init__(val)
        print "Mul5 Before:", self.val
        self.val *= 5
        print "Mul5", self.val

class Pro(Mul5, Add2):
    pass

class Incr(Pro):
    def __init__(self, val):
        super(Pro, self).__init__(val)
        self.val += 1
        print "Incr", self.val

p = Incr(5)
print(p.val)
```

> 答案

    输出是 36 ，具体可以参考 New-style Classes , multiple-inheritance

## Python 特殊方法

    描述
    我写了一个通过重载 new 方法来实现单例模式的类。

```python
class Singleton(object):
    _instance = None
    def __new__(self, *args, **kwargs):
        if self._instance:
            return self._instance
        self._instance = cv = object.__new__(self, *args, **kwargs)
        return cv

sin1 = Singleton()
sin2 = Singleton()
print(sin1 is sin2)
print Singleton() is sin2

# output: True
```


现在我有一堆类要实现为单例模式，所以我打算照葫芦画瓢写一个元类，这样可以让代码复用：

```python
class SingleMeta(type):
    def __init__(self, name, bases, dict):
        self._instance = None
        __new__o = self.__new__

        def __new__(self, *args, **kwargs):
            if self._instance:
                return self._instance
            self._instance = cv = __new__o(self, *args, **kwargs)
            return cv
        self.__new__ = __new__

class A(object):
    __metaclass__ = SingleMeta

a1 = A() # what`s the fuck
```

哎，为啥这会报错啊，我明明之前用这种方法给 `__getattribute__ `打补丁的，下面这段代码能够捕获一切属性调用并打印参数

```python
class TraceAttribute(type):
    def __init__(cls, name, bases, dict):
        __getattribute__o = cls.__getattribute__

        def __getattribute__(self, *args, **kwargs):
            print('__getattribute__:', args, kwargs)
            return __getattribute__o(self, *args, **kwargs)
        cls.__getattribute__ = __getattribute__

class A(object):  # Python 3 是 class A(object,metaclass=TraceAttribute):
    __metaclass__ = TraceAttribute
    a = 1
    b = 2
a = A()
a.a
a.b
```

a.b

试解释为什么给 getattribute 打补丁成功，而 new 打补丁失败。
如果我坚持使用元类给 new 打补丁来实现单例模式，应该怎么修改？

> 答案

其实这是最气人的一点，类里的`__new__`是一个 **staticmethod** 因此替换的时候必须以 **staticmethod** 进行替换。答案如下：

```python
class SingleMeta(type):
    def __init__(self, name, bases, dict):
        self._instance = None
        __new__o = self.__new__

        @staticmethod
        def __new__(self, *args, **kwargs):
            if self._instance:
                return self._instance
            self._instance = cv = __new__o(self, *args, **kwargs)
            return cv
        self.__new__ = __new__

class A(object):
    __metaclass__ = SingleMeta

a1 = A() # what`s the fuck
```
## 多线程是真的多线程吗？
> Effective Python 第37条：可以使用线程来执行阻塞式IO，但是不要用它做平行计算。

```python
#! /usr/bin/env python2.7
import threading
from time import sleep, ctime

loops = [4, 4]
class ThreadFunc(object):
    def __init__(self, func, args, name=''):
        self.name = name 
        self.func = func
        self.args = args 
    def __call__(self):
        apply(self.func, self.args)

def loop(nloop, nsec):
    print "start loop ", nloop, "at:", ctime()
    a = 0
    for i in xrange(30000000):
        a += i
    print a
    print "loop ", nloop, "done at:", ctime()


def main_with_muilt_thread():
    threads = []
    nloops = range(len(loops))
    for i in nloops:
        t = threading.Thread(target=ThreadFunc(loop,(i, loops[i]),loop.__name__))
        threads.append(t)
    print "starting at:", ctime()
    for i in nloops:
        threads[i].start()
    for i in nloops:
        threads[i].join()
    print "all DONE at:", ctime()
def main_with_one_thread():
    threads = []
    nloops = range(len(loops))
    for i in nloops:
        t = threading.Thread(target=ThreadFunc(loop,(i, loops[i]),loop.__name__))
        threads.append(t)
    print "starting at:",ctime()
    for i in nloops:
        loop(i, loops[i])
    print "all DONE at:",ctime()
if __name__ == '__main__':
    main_with_one_thread()
    main_with_muilt_thread()

```

> output

    starting at: Sun Jul 30 20:05:28 2017
    start loop  0 at: Sun Jul 30 20:05:28 2017
    449999985000000
    loop  0 done at: Sun Jul 30 20:05:30 2017
    start loop  1 at: Sun Jul 30 20:05:30 2017
    449999985000000
    loop  1 done at: Sun Jul 30 20:05:32 2017
    all DONE at: Sun Jul 30 20:05:32 2017
    starting at: Sun Jul 30 20:05:32 2017
    start loop  0 at: Sun Jul 30 20:05:32 2017start loop 
    1 at: Sun Jul 30 20:05:32 2017
    449999985000000
    loop  0 done at: Sun Jul 30 20:05:40 2017
    449999985000000
    loop  1 done at: Sun Jul 30 20:05:40 2017
    all DONE at: Sun Jul 30 20:05:40 2017

从中看出，使用单线程顺序执行时使用了4s，但是使用多线程执行了8s。本来使用多线程应该是原的两倍才对啊，但是现在多线程竟然比但想成还慢。因为**标准CPython解释器中的多线程受到了GIL的影响，同一时刻只能有一个线程得到执行**。但是为啥还要支持多线程呢？
1. 程序看上去可以同时执行很多个事情，免去了手工管理任务的切换操作
2. 处理阻塞式IO操作。Python执行某些系统调用时，会触发阻塞式操作。读写文件，在网络间通讯，显示器与设计之间交互都属于阻塞式IO。为了响应阻塞式请求，开发者可以借助线程，把python程序与这些耗时IO操作隔离开来。

## Python中一些忽视的性质

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


## 常见问题
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

## 参考文献

[1]. [Someone-tell-me-that-you-think-Python-is-simple]<https://manjusaka.itscoder.com/2016/11/18/Someone-tell-me-that-you-think-Python-is-simple/>

[2].[invoking-descriptors]<https://docs.python.org/2/reference/datamodel.html#invoking-descriptors>  

[3].[Descriptor HowTo Guide]<https://docs.python.org/3/howto/descriptor.html>

[4].[Python代码规范PEP 487]<https://www.python.org/dev/peps/pep-0487/#adding-a-class-attribute-with-the-attribute-order> 

[5].[what`s new in Python 3.6]<https://docs.python.org/3.6/whatsnew/3.6.html> 

[6].[Python相见恨晚的库]<https://www.zhihu.com/question/24590883>

[7]. [python recipes]<https://github.com/cwlseu/pyrecipes.git>