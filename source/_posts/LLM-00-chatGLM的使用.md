---
title: LLM-00-chatGLM的使用
date: 2023-05-06 16:14:54
tags:
---

其实按照官方的readme一步步执行即可。

首先clone项目 https://github.com/THUDM/ChatGLM-6B 
```shell
git clone https://github.com/THUDM/ChatGLM-6B.git
```

> 文章撰写时的commit_id为2873a6f452340565ff3cd130d5f7009a35c12154


安装依赖
```shell
pip install -r requirements.txt
```


使用transformer包即可下载huggingface的模型并使用，具体的code在官方的readme中。如果需要指定模型的版本，在可以在 `from_pretrained` 的调用中增加 `revision="v0.1.0"` 参数。

![](20230506161935.jpg)

此外还有api调用和gradio界面等等方式，这里不再赘述。总之，想要简单的用起来还是很容易的，readme写的很好。

本文作为LLM系列的开始，内容比较简单，后面会有新的探索会继续更新。