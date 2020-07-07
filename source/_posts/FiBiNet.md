---
title: FiBiNet：特征重要性+Bilinear交叉
date: 2020-07-07 23:40:01
tags: ['ML']
categories: ['ML','论文阅读','推荐系统']
---

FiBiNET: Combining Feature Importance and Bilinear feature Interaction for Click-Through Rate Prediction

## 简介
文章指出当前的许多通过特征组合进行CTR预估的工作主要使用特征向量的内积或哈达玛积来计算交叉特征，这种方法忽略了特征本身的重要程度。提出通过使用Squeeze-Excitation network (SENET) 结构动态学习特征的重要性以及使用一个双线性函数(Bilinear function)来更好的建模交叉特征。



## 方法
本文的网络结构图如下：
![](15941047971358.jpg)

可以看到有2处与普通的CTR模型不同：1.SENET Layer 2.Bilinear-Interaction Layer。我们着重看这两部分。假设ID特征有$f$个，第$i$个特征的embedding为$e_i \in R^k$，$k$为embed_size，下面先看一次前向的数据流：
1.$E=[e_1,e_2,...,e_f]$，$A$=SENET($E$)，A为每个ID特征的权重，$A\in R^f$
2.用权重$A$给$E$的对应特征加权，得到了加权后的Embedding $V=[v_1,...,v_f],v_i\in R^k$
3.$p$=Bilinear-Interaction($E$)，$p=[p_1,...,p_n]$，其中$n=\frac{f(f-2)}{2}$，即交叉组合数。同理$q$=Bilinear-Interaction($V$)
4.将$p$和$q$concat起来之后过DNN输出结果。

### SENET
#### 1.Squeeze
将每一个ID特征压缩到一个标量，压缩可以是mean-pooling或者max-pooling，下面是mean-pooling的数学表示
$$
z_i = F_{sq}(e_i) = \frac{1}{k}\sum_{t=1}^ke_i^{(t)}
$$
#### 2.Excitation
得到$z\in R^{f\times 1}$后，类似AutoEncoder，过两层全连接得到每个特征的权重$A=[a_1,...,a_f]$
$$
A=F_{e x}(Z)=\sigma_{2}\left(W_{2} \sigma_{1}\left(W_{1} Z\right)\right)
$$
其中$\sigma$为激活函数，$W_1\in R^{f\times f/r}$，$W_2\in R^{f/r \times f}$，$r$为缩减比例，据说设置为8比较好。
这一步我理解是把不重要的特征的weight进一步压低。
#### 3.Re-weight
最后一步即是用$A$缩放$E$得到$V$
$$
V=F_{R e W e i g h t}(A, E)=\left[a_{1} \cdot e_{1}, \cdots, a_{f} \cdot e_{f}\right]=\left[v_{1}, \cdots, v_{f}\right]
$$

### Bilinear-Interaction
一般的交叉方法用内积（$\cdot$）或者哈达玛积（$\odot$）
$$
\begin{aligned}
\left[a_{1}, a_{2}, \ldots, a_{n}\right] \cdot\left[b_{1}, b_{2}, \ldots, b_{n}\right] &=\sum_{i=1}^{n} a_{i} b_{i} \\
\left[a_{1}, a_{2}, \ldots, a_{n}\right] \odot\left[b_{1}, b_{2}, \ldots, b_{n}\right] &=\left[a_{1} b_{1}, a_{2} b_{2}, \ldots, a_{n} b_{n}\right]
\end{aligned}
$$
作者认为这些方法没有办法很好表示稀疏特征之间的组合，他选择在特征交叉时加入一个($k\times k$)的参数$W$
$$
p_{i,j} = v_i \cdot W \odot v_j
$$
这样要学习的$W$会暴增，作者提出了三种方案：
1. Field-ALL：所有的交叉公用一个$W$
2. Field-Each：每个在左边的$v_i$都有一个$W_i$，则一共需要$f$个$W$
3. Field-Interaction：每一对$(v_i,v_j)$单独一个$W$，一共需要$\frac{f(f-1)}{2}$个$W$

通过Bilinear-Interaction Layer，$E \to p, V \to q$，最终$[p,q]$进DNN输出最后的logits。