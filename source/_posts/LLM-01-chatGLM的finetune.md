---
title: LLM-01-chatGLM的finetune
date: 2023-05-06 16:22:05
tags:
---

在LLM大火的今天，无数技术研究者想要利用chatGPT提高自己的生产力。但由于其不开源，无法满足各种定制化的需求。在中文环境中，chatGLM是一个较为优秀的开源LLM模型。我们可以基于它进行微调，从而提升它在某些垂直领域的能力，满足定制化的需求。

> 本文主要参考chatGLM项目中的finetune指引：
> https://github.com/THUDM/ChatGLM-6B/tree/main/ptuning



## 01-环境准备

建议内存48G以上，显存24G以上。

python环境按照chatGLM项目的需求进行配置即可。

配置完成后，检查显卡是否可用
![](Pasted image 20230506113847.png)

## 02-模型准备

首先clone项目 https://github.com/THUDM/ChatGLM-6B 
```shell
git clone https://github.com/THUDM/ChatGLM-6B.git
```

> 文章撰写时的commit_id为2873a6f452340565ff3cd130d5f7009a35c12154
> hugging face上模型的commit_id为：658202d88ac4bb782b99e99ac3adff58b4d0b813

然后按照readme中《从本地加载模型》章节的指引，将模型下载至本地。本文将其放置在`ChatGLM-6B/model/chatglm-6b`目录下。

修改`cli_demo.py`中`from_pretrained`方法的路径，将模型路径改为我们刚才设置的路径下

通过运行cli_demo.py来测试是否一切OK，如果出现问题，建议逐一check下载文件的sha256是否跟hugging face上一致。

![](Pasted image 20230506130900.png)



## 03-训练数据

chatGLM官方的微调指引中，使用了[P-Tuning v2](https://github.com/THUDM/P-tuning-v2) 的微调技术。数据集使用的是 [ADGEN](https://aclanthology.org/D19-1321.pdf) (广告生成) 数据集。

ADGEN 数据集任务为根据输入（content）生成一段广告词（summary）。
```json
{
    "content": "类型#上衣*版型#宽松*版型#显瘦*图案#线条*衣样式#衬衫*衣袖型#泡泡袖*衣款式#抽绳",
    "summary": "这件衬衫的款式非常的宽松，利落的线条可以很好的隐藏身材上的小缺点，穿在身上有着很好的显瘦效果。领口装饰了一个可爱的抽绳，漂亮的绳结展现出了十足的个性，配合时尚的泡泡袖型，尽显女性甜美可爱的气息。"
}
```

从 [Google Drive](https://drive.google.com/file/d/13_vf0xRTQsyneRKdD1bZIr93vBGOczrk/view?usp=sharing) 或者 [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/b3f119a008264b1cabd1/?dl=1) 下载处理好的 ADGEN 数据集，将解压后的 `AdvertiseGen` 目录放到项目的目录下。

![-c200](Pasted image 20230506115651.png)

## 04-微调模型

还需要安装依赖
```shell
pip install jieba datasets rouge_chinese
```

进入`ptuning`文件夹，修改`train.sh`中数据集的路径和本地模型的路径，如果按照上图的结构放置训练数据集和模型，则需在路径前加`../`
其他配置可按需修改。运行p-tuning执行下面命令
```shell
bash train.sh
```

`quantization_bit`可以对模型进行量化，适合显存较小的情况。默认配置（quantization_bit=4、per_device_train_batch_size=1、gradient_accumulation_steps=16）的情况下训练仅需6个G的显存，训练3000个step需要4个多小时。
![](Pasted image 20230506132259.png)


如果不使用量化，其他参数不变的情况下，我这边实测需要13.5G的显存，但训练速度反而更快了。
![](Pasted image 20230506133044.png)



上述的默认配置中，除了量化之外的两个参数，表示一次训练迭代会以 1 的批处理大小进行 16 次累加的前后向传播，等效为 16 的总批处理大小。若想在同等批处理大小下提升训练效率，可在二者乘积不变的情况下，加大per_device_train_batch_size的值，但会增加显存消耗。我这里按照实际情况调整了参数，目前已经可以在1.5小时内完成训练了。

![-c500](Pasted image 20230506150312.png)


如果需要进行全参数的 Finetune，需要安装 [Deepspeed](https://github.com/microsoft/DeepSpeed)，然后运行以下指令：
```shell
bash ds_train_finetune.sh
```
由于资源问题（穷），这部分就先跳过了。之后有机会在详细介绍。


### 使用自己的数据集微调

修改 `train.sh` 和 `evaluate.sh` 中的 `train_file`、`validation_file`和`test_file`为你自己的 JSON 格式数据集路径，并将 `prompt_column` 和 `response_column` 改为 JSON 文件中输入文本和输出文本对应的 KEY。可能还需要增大 `max_source_length` 和 `max_target_length` 来匹配你自己的数据集中的最大输入输出长度。


### 对话数据集
如需要使用多轮对话数据对模型进行微调，可以提供聊天历史，例如以下是一个三轮对话的训练数据：
```json
{"prompt": "长城h3风扇不转。继电器好的。保险丝好的传感器新的风扇也新的这是为什么。就是继电器缺一个信号线", "response": "用电脑能读数据流吗？水温多少", "history": []}
{"prompt": "95", "response": "上下水管温差怎么样啊？空气是不是都排干净了呢？", "history": [["长城h3风扇不转。继电器好的。保险丝好的传感器新的风扇也新的这是为什么。就是继电器缺一个信号线", "用电脑能读数据流吗？水温多少"]]}
{"prompt": "是的。上下水管都好的", "response": "那就要检查线路了，一般风扇继电器是由电脑控制吸合的，如果电路存在断路，或者电脑坏了的话会出现继电器不吸合的情况！", "history": [["长城h3风扇不转。继电器好的。保险丝好的传感器新的风扇也新的这是为什么。就是继电器缺一个信号线", "用电脑能读数据流吗？水温多少"], ["95", "上下水管温差怎么样啊？空气是不是都排干净了呢？"]]}
```

训练时需要指定 `--history_column` 为数据中聊天历史的 key（在此例子中是 `history`），将自动把聊天历史拼接。要注意超过输入长度 `max_source_length` 的内容会被截断。

可以参考以下指令：
```shell
bash train_chat.sh
```


## 05-测试模型
文章撰写时的版本，保存的是新 Checkpoint（只包含 PrefixEncoder 参数），因此也需要load原有的模型参数。这里将上一级目录的`cli_demo.py`复制到ptuning目录下，并且将model.eval()之前的内容修改为
```python
import os
import platform
import signal
import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig

CHECKPOINT_PATH = "../output/adgen-chatglm-6b-pt-128-2e-2/checkpoint-3000"

tokenizer = AutoTokenizer.from_pretrained("../model/chatglm-6b", trust_remote_code=True)
config = AutoConfig.from_pretrained("../model/chatglm-6b", trust_remote_code=True, pre_seq_len=128)
model = AutoModel.from_pretrained("../model/chatglm-6b", config=config, trust_remote_code=True)
prefix_state_dict = torch.load(os.path.join(CHECKPOINT_PATH, "pytorch_model.bin"))
new_prefix_state_dict = {}
for k, v in prefix_state_dict.items():
    if k.startswith("transformer.prefix_encoder."):
        new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)

model = model.half().cuda()
model.transformer.prefix_encoder.float()
model = model.eval()
```

运行cli_demo.py后，就可以进行测试了。
![](Pasted image 20230506151424.png)

对比fine-tune前的模型
![](Pasted image 20230506151727.png)

可以看到在训练数据集上的回答效果显著提升


## 06-评估模型

评估模型是去评估微调模型的好坏，我们可以调用`evaluate.sh`来进行评估，其中部分路径也需要进行修改。评测指标为中文 Rouge score 和 BLEU-4，会将评估结果输出到文本文件中。

官方对比了全量微调，ptuning微调和lora微调的效果。其中LoRA实现采用的是 [simple_thu_chatglm6b](https://github.com/yuanzhoulvpi2017/zero_nlp/tree/main/simple_thu_chatglm6b)

结果如下图所示，在官方的测试中，p-tuning > Finetue > Lora
![-c500](Pasted image 20230506134755.png)

## 总结
本文主要记录了一次对于fine-tune的尝试。下一步可能会去了解一下langchain的细节，做一些有意思的东西。