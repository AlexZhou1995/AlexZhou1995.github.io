---
title: AutoInt：构造高阶特征交叉
date: 2020-07-07 23:34:56
tags: ['ML']
categories: ['ML','论文阅读','推荐系统']
---

AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks (CIKM 2019)

<!--more-->

## 简介

本文通过另外的角度来做特征交叉和构造高阶特征，结构如下图
![](15941030467753.jpg)


一句话总结：**把Dense特征也转化为embedding，然后所有特征一起过Multi-Head Attention。**

对dense特征做embedding：每个dense特征对应一个嵌入向量，乘以具体的dense特征值 作为其最终的emeddding。

## 优缺点
优点：简单易实现；可以让Dense和ID特征交叉。
缺点：提升似乎不是特别显著，如果模型里有了PNN，看起来再加AutoInt提升不大。