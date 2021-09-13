---
title: CapsuleNetwork调研
date: 2021-09-10 15:48:53
tags: ['ML']
categories: ['ML','深度学习']
---

## 为什么提出capsule
在讨论胶囊网络前，我们先来看一下目前最通用的深度学习模型之一，卷积神经网络（CNN）。
CNN目前已经完成了很多不可思议的任务，对于整个机器学习领域都产生了很大的影响。然而，CNN善于检测特征，却在探索特征（视角，大小，方位）之间的空间关系方面效果较差。举一个简单的例子，对于一张人脸而言，它的组成部分包括面部轮廓，两个眼睛，一个鼻子和一张嘴巴。对于CNN而言，这些部分就足以识别一张人脸；然而，这些组成部分的相对位置以及朝向就没有那么重要。
![](16086894305226.jpg)

<!--more-->

一个简单的CNN模型可以正确提取鼻子、眼睛和嘴巴的特征，但会错误地激活神经元进行人脸检测。如果不了解空间方向，大小不匹配，那么对于人脸检测的激活将会太高，比如下图95%。

![](16086894931180.jpg)

现在，假设每个神经元都包含特征的可能性和属性。例如，神经元输出的是一个包含 \[可能性，方向，大小\] 的向量。利用这种空间信息，就可以检测鼻子、眼睛和耳朵特征之间的方向和大小的一致性，因此对于人脸检测的激活输出就会低很多。

![](16086895133550.jpg)

而如果我们将神经元从标量升级为向量，则相同的胶囊就可以检测不同方向的同一个物体类别

![](16086896627419.jpg)


## Capsule的结构

基本一张图搞定，很好理解。这张图下游只有一个向量，实际上可以有多个向量，然后通过动态路由来决定每一个c的值。

![](16086902514866.jpg)


![](16086901663237.jpg)

唯一需要想明白的是，为什么要这样做squash

## 动态路由
动态路由确定的是上图的标量c，动态路由的motivation：低层的胶囊将会输入到和它“一致”的胶囊中。
> Lower level capsule will send its input to the higher level capsule that “agrees” with its input. This is the essence of the dynamic routing algorithm.

动态路由算法如下所示：
![](16086904139573.jpg)

也十分的好理解，刚开始将分配权重b设置为0，将l层的向量平均分发给l+1层，计算得出l+1层的结果后，反向计算l层向量到l+1层向量的距离，以此修改分配权重b。把l+1看做是聚类中心就更好理解了。

## MIND
MIND在capsule的基础之上，对其进行了一些改进。

> 胶囊是一种新的神经元，它由传统的神经网络用一个向量来表示，而不是用一个标量来表示。基于向量的胶囊预计能够代表一个实体的不同性质，其中胶囊的方向代表一个属性，胶囊的长度用来表示该属性存在的概率。相应地，多兴趣提取器层的目标是学习表示用户兴趣属性的表示以及是否存在相应的兴趣。胶囊与兴趣表征之间的语义联系促使我们将行为/兴趣表征视为行为/兴趣胶囊，并采用动态路径从行为胶囊中学习兴趣胶囊。然而，原有的图像数据路由算法并不能直接应用于用户行为数据的处理。因此，我们提出了行为兴趣（B2I）动态路由算法，将用户的行为自适应地聚合到兴趣表示向量中，与原有的路由算法在三个方面有所不同：Shared bilinear mapping matrix、Randomly initialized routing logits、Dynamic interest number.

### Shared bilinear mapping matrix
其实也是fixed bilinear mapping matrix，也就是上面结构图中的$W$，对于不同的capsule，只用同一个映射变量$W$。作者的理由：
1. 用户序列是变长的，因此不好设置某个数量的$W$
2. 希望item embedding映射到同一个空间

感觉有道理。

### Randomly initialized routing logits
顾名思义，高斯分布随机初始化的c，代替原本全0的c

### Dynamic interest number
对于每个用户，兴趣的个数是动态确定的：
$$
K_{u}^{\prime}=\max \left(1, \min \left(K, \log _{2}\left(\left|\mathcal{I}_{u}\right|\right)\right)\right)
$$
这条感觉是凑数的

## 代码
可以看 https://github.com/bojone/Capsule
或者这里也有另一个实现：
```python
import tensorflow as tf
from tensorflow.python.keras.initializers import RandomNormal
from tensorflow.python.keras.layers import Layer


class CapsuleLayer(Layer):
    def __init__(self, input_units, out_units, max_len, k_max, iteration_times=3,
                 init_std=1.0, **kwargs):
        self.input_units = input_units
        self.out_units = out_units
        self.max_len = max_len
        self.k_max = k_max
        self.iteration_times = iteration_times
        self.init_std = init_std
        super(CapsuleLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # self.routing_logits = self.add_weight(shape=[1, self.k_max, self.max_len],
        #                                       initializer=RandomNormal(stddev=self.init_std),
        #                                       trainable=False, name="B", dtype=tf.float32)
        self.bilinear_mapping_matrix = self.add_weight(shape=[self.input_units, self.out_units],
                                                       initializer=RandomNormal(stddev=self.init_std),
                                                       name="S", dtype=tf.float32)
        super(CapsuleLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        behavior_embddings, seq_len = inputs
        batch_size = tf.shape(behavior_embddings)[0]
        #seq_len = tf.squeeze(seq_len)
        seq_len_tile = tf.tile(seq_len, [1, self.k_max])

        routing_logits = tf.stop_gradient(tf.truncated_normal(shape=[1, self.k_max, self.max_len], stddev=self.init_std, name='B'))

        for i in range(self.iteration_times):
            mask = tf.sequence_mask(seq_len_tile, self.max_len)
            pad = tf.ones_like(mask, dtype=tf.float32) * (-2 ** 32 + 1)
            routing_logits_with_padding = tf.where(mask, tf.tile(routing_logits, [batch_size, 1, 1]), pad)
            weight = tf.nn.softmax(routing_logits_with_padding)
            behavior_embdding_mapping = tf.tensordot(behavior_embddings, self.bilinear_mapping_matrix, axes=1)
            Z = tf.matmul(weight, behavior_embdding_mapping)
            interest_capsules = squash(Z)
            delta_routing_logits = tf.reduce_sum(
                tf.matmul(interest_capsules, tf.transpose(behavior_embdding_mapping, perm=[0, 2, 1])),
                axis=0, keep_dims=True
            )
            routing_logits += delta_routing_logits
        interest_capsules = tf.reshape(interest_capsules, [-1, self.k_max, self.out_units])
        return interest_capsules

    def compute_output_shape(self, input_shape):
        return (None, self.k_max, self.out_units)

    def get_config(self, ):
        config = {'input_units': self.input_units, 'out_units': self.out_units, 'max_len': self.max_len,
                  'k_max': self.k_max, 'iteration_times': self.iteration_times, "init_std": self.init_std}
        base_config = super(CapsuleLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def squash(inputs):
    vec_squared_norm = tf.reduce_sum(tf.square(inputs), axis=-1, keep_dims=True)
    scalar_factor = vec_squared_norm / (1 + vec_squared_norm) / tf.sqrt(vec_squared_norm + 1e-8)
    vec_squashed = scalar_factor * inputs
    return vec_squashed
```
调用的时候
```python
high_capsule = CapsuleLayer(input_units=8,
                            out_units=8, max_len=50,
                            k_max=3)((hist_seq, seq_len))
## hist_seq [None, 50, 8]
## seq_len [None, 50]
```


## 存在的问题
有人通过实验指出，Capsule的Routing算法并不合理（包括一些改进的routing算法）:[Capsule Networks Need an Improved Routing Algorithm](http://proceedings.mlr.press/v101/paik19a/paik19a.pdf)。
文章中提到了一些Routing的方法，这里总结一下：
1. CapNet 《Dynamic routing between capsules》
![](16087948402135.jpg)
1. EMCaps 《Matrix capsules with em routing》
![](16087949130702.jpg)
1. OptimCaps 《An optimization view on dynamic routing between capsules》
![](16087949392584.jpg)
1. GroupCaps 《Group equivariant capsule
networks》
![](16087949594454.jpg)
1. AttnCaps 《Dynamic capsule
attention for visual question answering》
![](16087949936883.jpg)


## 参考文献
1. https://medium.com/ai%C2%B3-theory-practice-business/understanding-hintons-capsule-networks-part-iii-dynamic-routing-between-capsules-349f6d30418

2. https://zhuanlan.zhihu.com/p/67910276