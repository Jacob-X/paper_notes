## Prompt Learning



### 背景

NLP的下游任务

![image-20230115155044801](http://picture.jacobx.top/markdown/image-20230115155044801.png)



### 微调方式

1. **Bert和GPT1**，实际上都是对最后一层的一个token的representation，然后直接去接一个分类层，然后进行微调

2. **T5**，110亿的模型，decode的方式，把所有的任务都映射成了 Sequence to sequence的形式，在输入里面加入了简单的demonstrations，训练一个decoder来得到需要的token，训练得到的是token，而不是直接需要的最终结果

   + demonstration 把任务映射成了细粒度的，就不需要对每一个下游任务后面加一个分类的layer重新训练

   + T5还是需要微调的，在数据集上设计合适的demonstration，设计一些合适的标签

     ![image-20230115160456887](http://picture.jacobx.top/markdown/image-20230115160456887.png)

3. **GTP3**

   + 1750亿参数，是一个auto regressive的模型

   + 微调这样的模型是不现实的，论文中使用prompt来适配下游任务
   + 微调的时候实际上不会再更新任何参数了，使用一些prompt的description来提示模型生成tokens，就是后来所谓的few-shot和zero-shot的形式
   + Zero shot，通过语言来描述任务，让模型来预测答案，没有参数更新
   + ![image-20230115161047274](http://picture.jacobx.top/markdown/image-20230115161047274.png)
   + Few shot，语言描述任务，然后给少部分的任务样例，没有参数更新
   + ![image-20230115161033717](http://picture.jacobx.top/markdown/image-20230115161033717.png)



### 模型的趋势

1. #### 规模

   + In context learning work的原因至今没有解释

   + 趋势是模型越大，效果越好，理解能力和生成能力都更好

   + capacity足够大就可以不断去学习新的知识，有潜力去完成没有定义过的复杂任务

2. #### 微调
   
   + 难以计算更新全部的参数
   + 模型的规模太大，对每个下游任务都存储一个微调版本的模型的存储，是不可能的
   + 监督泛化不充分



### 高效的微调大模型

1. **task和data的角度**，使用 prompt learning（增加一些额外的上下文），通过增强few-shot learning 的能力，减少预训练和模型微调的gap
2. **优化微调方式的角度**，使用 delta tuning，通过小参数的优化来驱动大模型

![image-20230115164356734](http://picture.jacobx.top/markdown/image-20230115164356734.png)





### Prompt learning



#### fine tuning回顾

+ 使用PLM作为基础的encoder
+ 对特殊的任务，在模型的最后增加一层特殊的layer
+ 微调所有的参数
+ 在pre-training和fine-tuning之间是有很大的gap的，pre-train的时候是不知道模型要干啥的
+ ![image-20230115164824766](http://picture.jacobx.top/markdown/image-20230115164824766.png)



#### prompt-learning

+ 使用PLM作为基础的encoder
+ 增加一个额外的，与mask结合的文本描述（template）
+ 把训练得到的labels转化为文字的label（verbalizer），就是一个映射的过程，把模型生成的文本映射成任务最后需要的结果
+ 把pre-training和fine-tuning之间的gap减小了
+ ![image-20230115165335060](http://picture.jacobx.top/markdown/image-20230115165335060.png)
+ 一个语义分析的例子
+ ![image-20230115165613806](http://picture.jacobx.top/markdown/image-20230115165613806.png)



### 关于prompt learning 的一些思考



#### 预训练的模型

+ ==auto-regressive模型（GPT-1, GPT-2, GPT-3; OPT…）==，单向的，利用上文信息预测下文，适用于超级大的预训练模型，auto regressive prompt
+ ![image-20230115181427208](http://picture.jacobx.top/markdown/image-20230115181427208.png)
+ ==Masked Language Modeling==，掩码模型 (BERT, RoBERTa, DeBERTa)，自然语言理解的任务，完型填空式Prompt
+ ![image-20230115181515182](http://picture.jacobx.top/markdown/image-20230115181515182.png)
+ ==Encoder-Decoder架构的模型 (T5, BART)==
+ encoder是双向的attention机制
+ decoder是自回归的
+ ![image-20230115181709445](http://picture.jacobx.top/markdown/image-20230115181709445.png)



#### prompt模板的设计

+ 手工生成，hard prompt

+ 自动生成，soft prompt



#### Verbalizer的设计

+ 手工设计
+ 使用额外的知识库来扩充verbalizer的内容



#### Template



设计方式如下



==基于任务的特点来手动设计==，结合人类的先验知识
![image-20230115183317117](http://picture.jacobx.top/markdown/image-20230115183317117.png)



==通过搜索优化来自动生成==，抽取输入内容来生成template，自适应的方法，在zero-shot任务上的效果很好

Prompt-learning for fine-grained entity typing. 2021

![image-20230115183501367](http://picture.jacobx.top/markdown/image-20230115183501367.png)



==结合规则和逻辑生成template==，逻辑增强的模板

PTR: Prompt Tuning with Rules for Text Classification. 2021

![image-20230115183728566](http://picture.jacobx.top/markdown/image-20230115183728566.png)	



==结构化的，配合一些规则==

ProQA: Structural Prompt-based Pre-training for Unified Question Answering. 2021

+ 键值对的template
+ 将不同任务统一成一种结构化的形式
+ 本质上还是在提醒这个模型该干什么，虽然用一个统一的范式训练很多个任务，但是通过给不同的提示，让模型从内部对不同的任务进行一个区分，是不同维度上的区分
+ 用一个统一的范式训练许多不同的任务，这是prompt的独特之处

![image-20230115204129686](http://picture.jacobx.top/markdown/image-20230115204129686.png)	



==Ensembling Templates， 集成式的模板==

+ 对一个输入采用不同的模板

+ 减少了prompt engineering的开销

+ 稳定的模型的表现

  ![image-20230115210015214](http://picture.jacobx.top/markdown/image-20230115210015214.png)	

+ 把每个模板的结果进行结合，平均处理，加权平均



==自动搜索构建template==

AUTOPROMPT: Eliciting Knowledge from Language Models with Automatically Generated Prompts. 2020

+ 定义一些trigger tokens插入到template的里面，然后通过最大化后验概率的方法去优化prompt的embedding

+ 缺点：可解释性差

  ![image-20230115211134128](http://picture.jacobx.top/markdown/image-20230115211134128.png)



==使用encoder-decoder模型（T5）自动生成prompt==

LM-BFF: Making Pre-trained Language Models Better Few-shot Learners. 2021

![image-20230115211413790](http://picture.jacobx.top/markdown/image-20230115211413790.png)



### 对于soft pormpt的优化

+ P-tuning v1: prompts to the input layer (with Reparameterization)，对输入层加上prompt，经过重参数化处理之后的

+ P-tuning v2: prompts to every layer (like prefix-tuning)，在每一层上都加上prompt

  ![image-20230115211911757](http://picture.jacobx.top/markdown/image-20230115211911757.png)

+ prompt tuning，冻结模型的原始参数，对增加的额外的prompt的参数进行微调学习



### Verbalizer

+ 映射，将不固定的标签映射成需要模型输出的答案
+ 构建方式：手工构建，自动生成
  1. 使用人类的先验知识去手工构建
  2. 使用词，句来初始化标签词，通过同义词去扩大这个label
  3. 使用外部的知识库初始化,然后扩大label
  4. 将标签分解成许多token
  5. 虚拟token，然后优化label embedding



### knowledgeable  prompt

Knowledgeable prompt-tuning: Incorporating knowledge into prompt verbalizer for text classification. 2021

+ 首先找出label

+ 然后通过外部的知识库去扩充这个label 对应的词

  ![image-20230115212839323](http://picture.jacobx.top/markdown/image-20230115212839323.png)



### virtual token

Prototypical Verbalizer for Prompt-based Few-shot Tuning. 2021

+ 用一些虚拟的token代表label words
+ 将 [MASK] 标记的隐藏状态投射到嵌入空间并学习原型，在分类任务中，对于每一个分类的预测的mask的hidden state，在特征空间中对这些hidden state进行聚类，学一个原型出来
+ 学习到的原型构成了表达器并映射了 PLM 输出到相应的标签。

![image-20230115214235089](http://picture.jacobx.top/markdown/image-20230115214235089.png)





### learning strategy

==学习范式的进化过程==

1. Traditional: Learning from scratch
2. After BERT: Pre-training-then-fine-tuning
3. T5: Pre-training-then-fine-tuning with text-to-text format
4. GPT: Pre-training, then use prompt & in-context for zero- and few- shot

==prompt-learning 带来的新的学习范式==

1. 对于中小模型：Pre-training, prompting, optimizing all the parameters (middle-size models, few-shot setting) 
2. Pre-training, adding soft prompts, freezing the model and optimizing the prompt embeddings (delta tuning perspective)，预训练之后加soft prompt，然后冻结原始模型的参数，只去训练soft prompt的参数
3.   Pre-training with prompted data, zero-shot inference (Instruction tuning& T0)



###  Prompt-Tuning 

The Power of Scale for Parameter-Efficient Prompt Tuning, 2021.

+ 冻结预训练模型的参数
+ 在输入层增加soft prompt
+ 模型参数超过100亿的时候，与全参数微调的效果几乎相同
+ 参数高效化的微调方式，相对于原始的全参数微调
+ 但是对参数小模型，prompt-tuning 的性能不好
+ 收敛的速度慢

![image-20230115215628192](http://picture.jacobx.top/markdown/image-20230115215628192.png)



### Pre-trained Prompt Tuning

The Power of Scale for Parameter-Efficient Prompt Tuning, 2021

+ 在全数据的时候，prompt tuning和fine tuning的效果是相当的

+ 在少量数据的时候，prompt的泛化能力其实就较差了

+ 通过pre-trian较好的初始化prompt，提高prompt的泛化能力



### Fine-tuning with Prompted Dat

FINETUNED LANGUAGE MODELS ARE ZERO-SHOT LEARNERS. 2021

使用提示数据来进行微调，instruction tuning

+ 在1300亿参数的模型上训练60个有prompt的任务
+ 有效的提高了模型的zero-shot能力



### T0

Multitask Prompted Training Enables Zero-Shot Task Generalization. 2021

+ 使用人工编辑的prompt来训练encoder-decoder的模型，训练了35个任务，然后做zero-shot 的generalization
+ 在这个模型里面，同一个task可能会有不同的prompt

![image-20230115221035510](http://picture.jacobx.top/markdown/image-20230115221035510.png)

+ 在一些任务上做prompt learing，然后在没见过的任务上做zero-shot

![image-20230115221116487](http://picture.jacobx.top/markdown/image-20230115221116487.png)

  

带有position embedding的任务不适合使用prompt的方法



### 医学领域的应用

Clinical Prompt Learning with Frozen Language Models. 2021

一个医疗决策系统

![image-20230115221502488](http://picture.jacobx.top/markdown/image-20230115221502488.png)



### 视觉领域

CPT: Colorful Prompt Tuning for Pre-trained Vision-Language Models. 2021

多模态的学习方法，使用文本进行提示



### Prompt-Learning总结

1. prompt-learng需要考虑：预训练语言模型，下游任务，人类的先验知识
2. Template和Vebalizer的设计是十分重要的
3. 在低数据的任务上，prompt-learning有广阔的前景
4. prompt-learning 有广泛的应用



## Delta tuning



![image-20230115223601935](http://picture.jacobx.top/markdown/image-20230115223601935.png)



### delta-tuning：参数高效的微调方式

+ 更新少部分参数来驱动大模型
+ 冻结原始预训练模型的参数
+ 参数高效化在预训练模型的环境下有了实现的可能性，对模型的预训练得到了效果较好的初始化参数



### delta-tuning的种类：

Delta Tuning: A Comprehensive Study of Parameter Efficient Methods for Pre-trained Language Models, 2022.

+ Addition-based methods introduce extra trainable neural modules or  parameters that do not exist in the original model，增量式
+ Specification-based methods specify certain parameters in the original  model or process become trainable, while others frozen，指定式
+ Reparameterization-based methods reparameterize existing parameters  to a parameter-efficient form by transformation，重参数化式

![image-20230115224716798](http://picture.jacobx.top/markdown/image-20230115224716798.png)



### Addition-based，增量式的

==Parameter-Efficient Transfer Learning for NLP, 2019==

+ 在Transformer Layer中增加一些小的neural modules（adapters）

+ 只微调adapters，保持别的参数冻结

+ Adapters are down-projection and up-projection，向下映射和向上映射

  ![image-20230115225118813](http://picture.jacobx.top/markdown/image-20230115225118813.png)

+ 相当于只微调了0.5-8%的模型参数

![image-20230115225345564](http://picture.jacobx.top/markdown/image-20230115225345564.png)



==LST: Ladder Side-Tuning for Parameter and Memory Efficient Transfer Learning, 2019.==

+ 在模型的外面加一个adapter，反向传播的时候就不需要计算模型的参数，只优化外部的adapter参数，不经过模型的主干网络，减少了计算量

![image-20230115225753700](http://picture.jacobx.top/markdown/image-20230115225753700.png)





==Prefix-Tuning: Optimizing Continuous Prompts for Generation, 2021==

Prefix-Tuning

+ 在transformer的每一层上都增加了soft prompt
+ 只优化soft prompt

![image-20230115230143971](http://picture.jacobx.top/markdown/image-20230115230143971.png)



==The Power of Scale for Parameter-Efficient Prompt Tuning, 2021==

prompt-tuning

+ 在输入层上增加了soft prompt，简化版本的Prefix-Tuning

![image-20230115230249860](http://picture.jacobx.top/markdown/image-20230115230249860.png)



### Specification-based，指定式

==BitFit: Simple Parameter-efficient Fine-tuning for Transformer-based Masked Language-models, 2021==

BitFit

+ 只优化bias

![image-20230115230420317](http://picture.jacobx.top/markdown/image-20230115230420317.png)





==Exploring Low-dimensional Intrinsic Task Subspace via Prompt Tuning, 2021.==

Intrinsic Prompt Tuning

+ 把模型微调映射到低维的子空间里面
+ 在5维的子空间里面进行训练，训练得到的结果放缩到原来的参数中，在120个NLP任务上的效果都还不错
+ 在大模型的条件下，可能许多任务都有一个公共的子空间，在子空间里面训练的参数是较少的

![image-20230115231219921](http://picture.jacobx.top/markdown/image-20230115231219921.png)

![image-20230115231425592](http://picture.jacobx.top/markdown/image-20230115231425592.png)



==LoRA: Low-Rank Adaptation of Large Langauge Models, 2021.==
