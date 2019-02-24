---
layout: post
title: "tool：Log Analysis 可视化"
categories: [blog ]
tags: [linux开发]
description: 使用html5技术实现数据可视化，主要使用过ajax动态请求的技术，使用现成的数据可视化js框架echarts, 实现测试log的可视化。
---
{:toc}

## Javascript框架调研

组里说要做一个操作系统不同版本的测试log的管理工具，我就先调研了一下如何实现测试log的可视化。原来在最近几年js发展的非常惊人，[这里有一些好的可视化框架](http://www.36dsj.com/archives/19522)，大家拿走不谢。

```js
var myChart;
var eCharts;

require.config({
	paths : {
		'echarts' : '${pageContext.request.contextPath}/js/echarts2.0/echarts' ,
		'echarts/chart/line' : '${pageContext.request.contextPath}/js/echarts2.0/echarts' //需要的组件
	}
});

require(
	[ 'echarts', 
		'echarts/chart/line'
	], DrawEChart //异步加载的回调函数绘制图表
);

//创建ECharts图表方法
function DrawEChart(ec) {
	eCharts = ec;
	myChart = eCharts.init(document.getElementById('main'));
	myChart.showLoading({
		text : "图表数据正在努力加载..."
	});
	//定义图表options
	var options = {
		title : {
			text : "未来一周气温变化",
			subtext : "纯属虚构",
			sublink : "http://www.baidu.com"
		},
		tooltip : {
			trigger : 'axis'
		},
		legend : {
			data : [ "最高气温" ]
		},
		toolbox : {
			show : true,
			feature : {
				mark : {
					show : true
				},
				dataView : {
					show : true,
					readOnly : false
				},
				magicType : {
					show : true,
					type : [ 'line', 'bar' ]
				},
				restore : {
					show : true
				},
				saveAsImage : {
					show : true
				}
			}
		},
		calculable : true,
		xAxis : [ {
			type : 'category',
			boundaryGap : false,
			data : [ '1', '2', '3', '4', '5', '6', '7' ]
		} ],
		yAxis : [ {
			type : 'value',
			axisLabel : {
				formatter : '{value} °C'
			},
			splitArea : {
				show : true
			}
		} ],
		grid : {
			width : '90%'
		},
		series : [ {
			name : '最高气温',
			type : 'line',
			data : [ 11, 22, 33, 44, 55, 33, 44 ],//必须是Integer类型的,String计算平均值会出错
			markPoint : {
				data : [ {
					type : 'max',
					name : '最大值'
				}, {
					type : 'min',
					name : '最小值'
				} ]
			},
			markLine : {
				data : [ {
					type : 'average',
					name : '平均值'
				} ]
			}
		} ]
	};
	myChart.setOption(options); //先把可选项注入myChart中
	myChart.hideLoading();
	getChartData();//aja后台交互 
}

function getChartData() {
	//获得图表的options对象
	var options = myChart.getOption();
	//通过Ajax获取数据
	$.ajax({
		type : "post",
		async : false, //同步执行
		url : "${pageContext.request.contextPath}/echarts/line_data",
		data : {},
		dataType : "json", //返回数据形式为json
		success : function(result) {
			if (result) {
				options.legend.data = result.legend;
				options.xAxis[0].data = result.category;
				options.series[0].data = result.series[0].data;

				myChart.hideLoading();
				myChart.setOption(options);
			}
		},
		error : function(errorMsg) {
			alert("不好意思，大爷，图表请求数据失败啦!");
			myChart.hideLoading();
		}
	});
}

```

后台代码，从网上找了一个javaweb的代码，先放这里，后面实现了Python的代码再来补充。

```java
@Controller
@RequestMapping("/echarts")
public class EntityController {
	
	private static final Logger logger = LoggerFactory.getLogger(EntityController.class);

	@RequestMapping("/line_data")
	@ResponseBody
	public EchartData lineData() {
		logger.info("lineData....");
		
		List<String> legend = new ArrayList<String>(Arrays.asList(new String[]{"最高气温"}));//数据分组
		List<String> category = new ArrayList<String>(Arrays.asList(new String []{"周一","周二","周三","周四","周五","周六","周日"}));//横坐标
		List<Series> series = new ArrayList<Series>();//纵坐标
		
		series.add(new Series("最高气温", "line", 
						new ArrayList<Integer>(Arrays.asList(
								21,23,28,26,21,33,44))));
		
		EchartData data=new EchartData(legend, category, series);
		return data;
	}
	
	@RequestMapping("/line_page")
	public String linePage() {
		logger.info("linePage....");
		return "report/line";
	}
	
	
}
```

## 实现计划
当前我们的组的数据主要是使用mongodb进行存储的。要不要考虑使用使用[django](http://www.ibm.com/developerworks/library/os-django-mongo/)呢，如果只是实现可视化这个任务，没有必要使用django框架，完全可以使用python(PyMongo+ echarts-python)+ MongoDB的解决方案就好了。但是如果考虑的后期的拓展或者应用的进一步增加的话，使用框架是一个比较好的选择。

* MongoDB + Echarts
* MongoDB + Django + Echarts

## Demo

首先安装pip和使用pip install echarts-python
然后直接运行下面例子：

```python
from echarts import Echart, Legend, Bar, Axis

chart = Echart('GDP', 'This is a fake chart')
chart.use(Bar('China', [2, 3, 4, 5, 1 ,2 ,3, 4, 25, 36, 27]))
chart.use(Legend(['GDP']))
chart.use(Axis('category', 'bottom', data=['Nov', 'Dec', 'Jan', 'Feb','March','April','Jun','Jul','Aug','Sep','Oct']))
chart.use(Axis('value','left', data= range(0, 100, 10 )))
chart.plot()
```
就在浏览器中出现了如下图片：
![图片](https://cwlseu.github.io/images/visualdata/bar.jpg)

## 参考链接

1. [echarts github地址](https://github.com/ecomfe/echarts)
2. [echarts 官网](http://echarts.baidu.com/index.html)
其中包含各种样例和API地址
3. [echarts python下载](https://github.com/yufeiminds/echarts-python)
4. [MongoDB 文档和中文社区](http://docs.mongoing.com/manual-zh/)
5. [Django + MongoDB IBM Blog](http://www.ibm.com/developerworks/library/os-django-mongo/)