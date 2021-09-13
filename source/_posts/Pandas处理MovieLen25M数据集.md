---
title: Pandas处理MovieLen25M数据集
date: 2020-03-21 21:09:19
tags: ['ML']
categories: ['ML','python']
---

最近做了一些推荐侧召回的实验尝试，除了在业务的数据上进行测试，我还想在公开数据集上进行验证。参考Related works，最终选择使用MovieLen和Amazon的数据集。由于这两个数据集给的是裸的数据，因此需要我们根据自己的需要做一些处理。这里我用pandas来做数据的分析和处理。

<!--more-->

Pandas是数据科学常用的工具，但是说来惭愧，我之前倒是很少用Pandas。因为之前做强化学习较多，不太需要大量的这种数据分析。因此这次也就趁机在熟悉一下。我个人觉得相比于算法原理，工具类的东西大致记录一下就好。工作中用到多了自然就记住了，记不住的说明平时也不怎么用，到时候再查就好，相关的文档网上有很多很多，倒是不必死记。

废话不多说，直接上Notebook吧，该说的都在注释里。（下面的链接是个html，但是如果我以HTML的格式放到资源文件夹里，编译后blog首页排版会有错）
{% asset_link ml_data_process [pandas数据处理notebook] %}

经过这样处理，我得到了几个处理后的表，表的含义见notebook
{% codeblock %}
viewed_sequence
genome_feature
genres_tag_one_hot_feature
movie_rating_feature
tag_feature_movie
tag_feature_user
user_rating_feature
{% endcodeblock %}

最后通过python+MPI并行的聚合生成最后的数据，写入到本地文件
{% asset_link run.py [python+MPI产出聚合数据集] %}

不过最终处理出来的数据集有十几个G，受限于线上docker容器的内存限制，无法放到一个文件里序列化存储和恢复，还是得通过file的方式来读，算是没有完美的达到预期，但是问题不大。