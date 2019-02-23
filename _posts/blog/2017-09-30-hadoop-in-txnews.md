---
layout: post
title: Hadoop到Spark的踩坑
categories: [blog ]
tags: [开发]
description: Spark和hadoop的踩坑之路
--- 
{:toc}
- 声明：本博客欢迎转发，但请保留原作者信息!
- 作者: [曹文龙]
- 博客： <https://cwlseu.github.io/> 

## install jdk in centos 

1. wget --no-cookies --no-check-certificate --header "Cookie:oraclelicense=accept-securebackup-cookie" "http://download.oracle.com/otn-pub/java/jdk/8u91-b14/jdk-8u91-linux-x64.rpm"

2. yum localinstall jdk-8u91-linux-x64.rpm

## 查看开发端口

netstat -nlp | grep tcp

## install Hadoop

## 重新启动hadoop之后

bin/hdfs namenode -format


## 运行Demo

下面看[MapReduce](http://hadoop.apache.org/docs/r2.7.3/hadoop-mapreduce-client/hadoop-mapreduce-client-core/MapReduceTutorial.html)的例子

```java
package com.tencent.omg;

import java.io.IOException;
import java.util.StringTokenizer;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class WordMain {

  public static class TokenizerMapper
       extends Mapper<Object, Text, Text, IntWritable>{

    private final static IntWritable one = new IntWritable(1);
    private Text word = new Text();

    public void map(Object key, Text value, Context context
                    ) throws IOException, InterruptedException {
      StringTokenizer itr = new StringTokenizer(value.toString());
      while (itr.hasMoreTokens()) {
        word.set(itr.nextToken());
        context.write(word, one);
      }
    }
  }

  public static class IntSumReducer
       extends Reducer<Text,IntWritable,Text,IntWritable> {
    private IntWritable result = new IntWritable();

    public void reduce(Text key, Iterable<IntWritable> values,
                       Context context
                       ) throws IOException, InterruptedException {
      int sum = 0;
      for (IntWritable val : values) {
        sum += val.get();
      }
      result.set(sum);
      context.write(key, result);
    }
  }

  public static void main(String[] args) throws Exception {
    // Configuration 类: 读取hadoop的配置文件，如 site-core.xml...;
    //也可以用set方法重新设置(会覆盖): conf.set("fs.defaultFS","hdfs://masterhost:9000")
    Configuration conf = new Configuration();
    // 新建一个job,传入配置信息
    Job job = Job.getInstance(conf, "word count");
    //设置主类
    job.setJarByClass(WordMain.class);
    //设置Mapper类
    job.setMapperClass(TokenizerMapper.class);
    // the output of each map is passed through the local combiner (which is same as the Reducer as per the job configuration) for local aggregation, after being sorted on the *key*s.
    job.setCombinerClass(IntSumReducer.class);
    // 
    job.setReducerClass(IntSumReducer.class);
    // 设置输出类型
    job.setOutputKeyClass(Text.class);
    job.setOutputValueClass(IntWritable.class);
    // 获取输入参数
    FileInputFormat.addInputPath(job, new Path(args[0]));
    FileOutputFormat.setOutputPath(job, new Path(args[1]));
    System.exit(job.waitForCompletion(true) ? 0 : 1);
  }
}
```

运行测试程序

`hadoop jar wordcount.jar com.tencent.omg.WordMain /user/root/input /user/root/out`
其中`hadoop jar`就是要运行jar的意思，接下来`wordcount.jar`是输入的jarFile，然后是[mainClass] 及mainClass的参数们 args...；
其中本程序中需要输入两个参数


## 出现的问题

> Datanode不能够启动的问题

根据[there-are-0-datanodes-running-and-no-nodes-are-excluded-in-this-operation](http://stackoverflow.com/questions/26545524/there-are-0-datanodes-running-and-no-nodes-are-excluded-in-this-operation)
- remove all temporary files ( by default in /tmp) - sudo rm -R /tmp/*.
- Now try connecting to all nodes through ssh by using ssh username@host and add keys in your master using ssh-copy-id -i ~/.ssh/id_rsa.pub username@host to give unrestricted access of slaves to the master (not doing so might be the problem for refusing connections). 
- Format the namenode using hadoop namenode -format and try restarting the daemons.

> 文件不存在问题

`bin/hdfs dfs -mkdir -p /user/${username}/output`

总体上来说，hdfs命令就是在原来的shell命令之前添加`hadoop dfs -`就是了，其他的就丢给文件系统去做显示，查找的了。

> 删除文件夹

/path/to/hadoop dfs -rm -rf /user/hadoop/topics/dumps

[hdfs shell官方文档](http://hadoop.apache.org/docs/r1.0.4/cn/hdfs_shell.html)中提及的一些命令已经基本够用了

## 向spark提交任务

首先将spark程序export为jar包

`. /data/pac_hadoop/spark/bin/spark-submit --class com.tencent.Lda --master spark://master:portnum ./xxxx/lda.jar  /user/hadoop/topics/dumps  /user/hadoop/model_path 1 1 30 > ./xxxx/segment_all_1.log`

最后的结果输出，然后根据
[Spark start](http://spark.apache.org/docs/latest/quick-start.html)
[spark官方文档翻译博客，可供参考](http://www.cnblogs.com/BYRans/p/5292763.html)

## reference

1. [Hadoop Commands Guide](http://hadoop.apache.org/docs/r2.7.3/hadoop-project-dist/hadoop-common/CommandsManual.html)
2. [本地编辑，hadoopserver部署运行案例-详细](https://freeshow.github.io/BigData/Hadoop/Windows%E4%B8%8B%E4%BD%BF%E7%94%A8eclipse%E7%BC%96%E8%AF%91%E6%89%93%E5%8C%85%E8%BF%90%E8%A1%8C%E8%87%AA%E5%B7%B1%E7%9A%84MapReduce%E7%A8%8B%E5%BA%8F%20Hadoop2.6.0/)
3. [一个python写的hadoop map reduce程序](https://github.com/hustlijian/hadoop-tutorial)


## 理论部分

LDA是一个主题模型，它能够推理出一个文本文档集合的主题。LDA可以认为是一个聚类算法，原因如下：
* 主题对应聚类中心，文档对应数据集中的样本（数据行）
* 主题和文档都在一个特征空间中，其特征向量是词频向量
* 跟使用传统的距离来评估聚类不一样的是，LDA使用评估方式是一个函数，该函数基于文档如何生成的统计模型。

LDA以**词频向量**表示的文档集合作为输入，输出结果提供：
* Topics：推断出的主题，每个主题是单词上的概率分布。
* Topic distributions for documents：对训练集中的每个文档，LDA给一个在主题上的概率分布。

## 查看spark集群

在master 节点上使用 `w3m http://localhost:8080`

问题：
1. LDA模型的训练集合哪里来， rcd_articles还是pac_articles
发现segment_all中的数据是从pac_articles
因为后面要去读取对应的docid的title

2.ssh连接断掉的原因是什么

/data/pac_hadoop/spark/bin/spark-submit --executor-cores 1   --executor-memory 1g  --num-executors 2 --class KafkaWordCount --master spark:// xxxx.xxxx.xxx.xxx:7077 xw_personas_proj-1.0.jar  xxxx.xxxx.xxx.xxx:9092,xxxx.xxxx.xxx.xxx:9092,xxxx.xxxx.xxx.xxx:9092 xw_user_logs

## 执行jar-> spark
/data/pac_hadoop/spark/bin/spark-submit --executor-cores 1   --executor-memory 1g  --num-executors 6 --class KafkaWordCount --master spark://xxxx.xxxx.xxx.xxx:7077 xw_personas_proj-1.0.jar xxxx.xxxx.xxx.xxx:9092 xw_user_logs

从remove拉取文件
scp hadoop@xx.xxx.xx.xx:/home/hadoop/xxxx lda_result_500_20

坑：
1. 上传下载数据，路径使用绝对路径，绝对路径，绝对路径重要的事情说三遍；尤其是使用python os.system()执行的时候。
2. 文件不要以_开头，下划线开头的文件在spark中表示隐藏文件，根本就不读取其中的内容。很多情况下以下划线开头表示隐藏文件
3. Scala是也是可以访问数据库的，技术选型很重要

SQL语句：不要有多余的空格
```UPDATE cwl_rcd_corpus_info_tx SET topic = '187:0.029912,107:0.027155,142:0.025944,10:0.021934,148:0.016860', update_time_tx=14934342342 WHERE article_id_tx=20170614034267;```

#!/bin/sh
export PATH=$PATH:/data/Python-2.7.3/bin
export PYTHONPATH=$PYTHONPATH:/data/pac_hadoop/spark/python

virtualenv topics
source topics/bin/activate

<!-- pip install scikit-learn -proxy=http://dev-proxy.oa.com:8080
pip install virtualenv -i http://mirror-sng.oa.com/pypi/web/simple/ --trusted-host mirror-sng.oa.com  -->


`/bin/sh /data/pac_hadoop/spark/bin/spark-submit --class com.tencent.TopicShow --master spark://xxxx.xxxx.xxx.xxx:7077 topicshow.jar`

`select pac_articles.article_id, title, content from pac_articles left join rcd_corpus_info on rcd_corpus_info.article_id where pac_articles.create_time between '1499244660' and  '1499307191' limit 10;`


`select count(a.article_id) from rcd_corpus_info b, pac_articles a where a.article_id=b.article_id and b.create_time between 1499137016 and 1499309816 order by b.create_time desc limit 30000 offset 0;`

`ps -ef | grep "topics_corpus_update_hdfs.py" | awk '{print $2}'|xargs kill -9`

## ERROR
```
SLF4J: Class path contains multiple SLF4J bindings.
SLF4J: Found binding in [jar:file:/data/pac_hadoop/spark/jars/slf4j-log4j12-1.7.16.jar!/org/slf4j/impl/StaticLoggerBinder.class]
SLF4J: Found binding in [jar:file:/data/pac_hadoop/spark/jars/slf4j-log4j12-1.6.1.jar!/org/slf4j/impl/StaticLoggerBinder.class]
SLF4J: See http://www.slf4j.org/codes.html#multiple_bindings for an explanation.
SLF4J: Actual binding is of type [org.slf4j.impl.Log4jLoggerFactory]
[Stage 551:======================================>                  (2 + 1) / 3]
Exception in thread "main" org.apache.spark.SparkException: Job aborted due to stage failure: Task 1 in stage 551.0 failed 4 times, 
most recent failure: Lost task 1.3 in stage 551.0 (TID 230, 10.185.25.224, executor 0): 
ExecutorLostFailure (executor 0 exited caused by one of the running tasks) 
Reason: Remote RPC client disassociated. Likely due to containers exceeding thresholds, or network issues. Check driver logs for WARN messages.
Driver stacktrace:
  at org.apache.spark.scheduler.DAGScheduler.org$apache$spark$scheduler$DAGScheduler$$failJobAndIndependentStages(DAGScheduler.scala:1435)
  at org.apache.spark.scheduler.DAGScheduler$$anonfun$abortStage$1.apply(DAGScheduler.scala:1423)
  at org.apache.spark.scheduler.DAGScheduler$$anonfun$abortStage$1.apply(DAGScheduler.scala:1422)
  at scala.collection.mutable.ResizableArray$class.foreach(ResizableArray.scala:59)
  at scala.collection.mutable.ArrayBuffer.foreach(ArrayBuffer.scala:48)
  at org.apache.spark.scheduler.DAGScheduler.abortStage(DAGScheduler.scala:1422)
  at org.apache.spark.scheduler.DAGScheduler$$anonfun$handleTaskSetFailed$1.apply(DAGScheduler.scala:802)
  at org.apache.spark.scheduler.DAGScheduler$$anonfun$handleTaskSetFailed$1.apply(DAGScheduler.scala:802)
  at scala.Option.foreach(Option.scala:257)
  at org.apache.spark.scheduler.DAGScheduler.handleTaskSetFailed(DAGScheduler.scala:802)
  at org.apache.spark.scheduler.DAGSchedulerEventProcessLoop.doOnReceive(DAGScheduler.scala:1650)
  at org.apache.spark.scheduler.DAGSchedulerEventProcessLoop.onReceive(DAGScheduler.scala:1605)
  at org.apache.spark.scheduler.DAGSchedulerEventProcessLoop.onReceive(DAGScheduler.scala:1594)
  at org.apache.spark.util.EventLoop$$anon$1.run(EventLoop.scala:48)
  at org.apache.spark.scheduler.DAGScheduler.runJob(DAGScheduler.scala:628)
  at org.apache.spark.SparkContext.runJob(SparkContext.scala:1925)
  at org.apache.spark.SparkContext.runJob(SparkContext.scala:1988)
  at org.apache.spark.rdd.RDD$$anonfun$fold$1.apply(RDD.scala:1089)
  at org.apache.spark.rdd.RDDOperationScope$.withScope(RDDOperationScope.scala:151)
  at org.apache.spark.rdd.RDDOperationScope$.withScope(RDDOperationScope.scala:112)
  at org.apache.spark.rdd.RDD.withScope(RDD.scala:362)
  at org.apache.spark.rdd.RDD.fold(RDD.scala:1083)
  at org.apache.spark.mllib.clustering.EMLDAOptimizer.computeGlobalTopicTotals(LDAOptimizer.scala:229)
  at org.apache.spark.mllib.clustering.EMLDAOptimizer.next(LDAOptimizer.scala:216)
  at org.apache.spark.mllib.clustering.EMLDAOptimizer.next(LDAOptimizer.scala:80)
  at org.apache.spark.mllib.clustering.LDA.run(LDA.scala:334)
  at com.tencent.Lda$.main(lda.scala:78)
  at com.tencent.Lda.main(lda.scala)
  at sun.reflect.NativeMethodAccessorImpl.invoke0(Native Method)
  at sun.reflect.NativeMethodAccessorImpl.invoke(NativeMethodAccessorImpl.java:62)
  at sun.reflect.DelegatingMethodAccessorImpl.invoke(DelegatingMethodAccessorImpl.java:43)
  at java.lang.reflect.Method.invoke(Method.java:498)
  at org.apache.spark.deploy.SparkSubmit$.org$apache$spark$deploy$SparkSubmit$$runMain(SparkSubmit.scala:743)
  at org.apache.spark.deploy.SparkSubmit$.doRunMain$1(SparkSubmit.scala:187)
  at org.apache.spark.deploy.SparkSubmit$.submit(SparkSubmit.scala:212)
  at org.apache.spark.deploy.SparkSubmit$.main(SparkSubmit.scala:126)
  at org.apache.spark.deploy.SparkSubmit.main(SparkSubmit.scala)
```