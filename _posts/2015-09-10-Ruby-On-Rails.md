https://ihower.tw/rails4/

## Ruby
* 动态类型
* 支持单继承 和 多态
* 每个操作都是方法，EveryThing is Object.
* DataType： array string 

### Build from source

Prepare: download source code from gov-net
1. install openssl

```shell
	$sudo apt-get install openssl 
	$sudo apt-get install libssl-dev
```
如果不做这一步，后面使用gem会报错：

	ERROR:  While executing gem ... (Gem::Exception)
    Unable to require openssl, install OpenSSL and rebuild ruby (preferred) or use non-HTTPS sources

更多信息，参考Stackoverflow[Gem couldn't require openssl](http://stackoverflow.com/questions/21201493/couldnt-require-openssl-in-ruby)
2. build ruby source code

	* 设置 --with-openssl 参数，否则不能够使用gem进行安装某些lib
	$./configure --with-openssl-dir=/usr/local/ssl
	* make and install 
	$make
	$make install

If some error occur, try `sudo`(System Permission)

3. install rails using the GEM

	$gem install rails -v 4.2.3
更多GEM操作：
	* 列出所有gem source
	$gem sources -l
	* 移除一个gem 源
	$gem sources -r https://rubygems.org/
	$gem sources -a https://ruby.taobao.org/

## javascript
不是语言本身可以做而是浏览器支持什么
浏览器兼容性问题是javascript中常见的问题

## CodeGrader
一个自动判分系统


* Agile Development & Scrum
* Plan and Document based Developement
* DRY
* MVC SaaS
* SMART 
* TDD and Red-Green-Refactor
* Lo-Fi UI 
* FIRST For test should:fast Indenpendent repeatable self-checking timely

Figure in the SAAS book
* Figure 1.7 
* Figure 2.2 the MVC relationship
* Figure 2.7 the 3-Tier 
主要是第一二章中的架构相关的图片


