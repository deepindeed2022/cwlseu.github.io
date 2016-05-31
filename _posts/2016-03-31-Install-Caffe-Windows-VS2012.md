



在window上使用caffe深度学习框架，安装路程艰辛，不过也是很有乐趣的。
## NuGet
[install](http://docs.nuget.org/consume/installing-nuget)
第一次接触到NuGet工具，很是帅气，简单一句话，就像是python里的pip，ubuntu里的sudo apt-get 命令，NuGet有一个server管理着大量的package，我们通过一个简单的*Install-Package* 的命令就可以实现对响应的依赖库的安装，很是方便。

## VS2012
vs2012由于对于C++11的支持还是不够全面的，在caffe中用了很多C++11的特性，导致错误。[MSVC对 C++11 Core Language Feature的支持性](https://msdn.microsoft.com/en-us/library/hh567368(v=vs.110).aspx)

'''
    D:\Document\Repo\VS2012\caffe\include\caffe/common.hpp(84): error : namespace "std" has no member "isnan"
    D:\Document\Repo\VS2012\caffe\include\caffe/common.hpp(85): error : namespace "std" has no member "isinf"
'''

## VS2013 
VS2013对于C++11的支持性就好多了

"pyconfig.h"或者"patchlevel.h"文件找不到的问题：
将python的安装路径下的include路径添加到项目include项目中。
如：C:\Develop\Python27\include


错误提示：error C2220: 警告被视为错误 - 没有生成“object”文件
错误原因：原因是该文件的代码页为英文，而我们系统中的代码页为中文。

解决方法：
1，将源码转化为正确的编码方式
    用vs2013打开对应的文档，文件->打开->选择该cpp，然后保存。
    如果不起作用的话，修改其中一部分，或者 选择替换，选中正则表达式，将\n替换为\n。
   也可以用文本编辑器如Notepad，更改代码文件的编码方式，改为ANSI。

2，设置项目属性，取消警告视为错误
    VS2013菜单 - 项目 - 属性 - 通用配置 - C/C++ - 常规 - 将警告视为错误 修改为 否，重新编译即可。


http://zhidao.baidu.com/link?url=rikLum87Ilmdo15lb2DWIDkM0P0d6UbE6BcZq1oJfXnA7B5C2EhptRUkuJgLmw_YJSByUizGU-xQe5nniFYaY_