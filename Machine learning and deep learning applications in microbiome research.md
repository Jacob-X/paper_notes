### 机器学习的在微生物研究的应用

+ 表型(即预测环境或宿主表型)
+ 微生物特征分类(即确定微生物组的丰度、多样性或分布)
+ 研究微生物组各组成部分之间的复杂物理和化学相互作用
+ 监测微生物组组成的变化等任务。



### 常用的机器学习方法

+ 线性回归，lasso and elastic nets
+ 随机森林
+ 支持向量机（一般不用，只做基准测试）



### 数据降维的方法

+ PCA，主成分分析
+ PCoA，主坐标分析
+ t-SNE，T随机领域嵌入
+ UMAP，均匀流形近似与投影



### 深度学习的方法

+ FCNN
+ RNN, 文献：Human host status inference from temporal microbiome changes via recurrent neural networks，使用纵向微生物组数据作为输入来推断人类宿主的状态
+ Autoencoder,  文献：
  + DeepMicro: deep representation learning for disease prediction based on microbiome data，DeepMicro 使用各种自动编码器成功地将高维微生物组数据转换为稳健的低维表示，并对学习到的表示应用机器学习分类算法。
  + Using Autoencoders for Predicting Latent Microbiome Community Shifts Responding to Dietary Changes，提出了一个深度神经网络来预测肠道微生物群对饮食变化的反应。 使用自动编码器来捕获数据中的内在结构，并使用人工神经网络来模拟微生物组的非线性动力学
  + Microbiome-based disease prediction with multimodal variational information bottlenecks，多模态变分信息瓶颈 (MVIB)，这是一种能够学习多种异构数据模式的联合表示的新型深度学习模型，依赖于专家乘积方法来集成来自两个自动编码器的信息，每个自动编码器的专家分别具有不同的形态：丰度(物种级别)和存在(应变级别)特征。
  + Learning representations of microbe–metabolite interactions，使用神经网络来估计在特定微生物存在的情况下每个分子存在的条件概率，发现微生物产生的代谢物与炎症性肠病之间的关系，通过迭代训练，mmvec（提出的模型名称，基于word2vec） 可以学习微生物和代谢物之间的共现概率
  + Decoding the language of microbiomes using word embedding techniques, and applications in inflammatory bowel disease，应用嵌入算法来量化来自美国肠道计划 (AGP) 微生物组众包工作的 18,000 多个样本中的分类单元共现模式，从而得出微生物组级别的属性。 比较使用属性、归一化分类计数数据和另一种常用的降维方法训练的模型的预测能力，主成分分析对炎症性肠病 (IBD) 患者和健康对照的样本进行分类。



### 常见的问题

+ 可解释性

  + Deep in the Bowel: Highly Interpretable Neural Encoder-Decoder Networks Predict Gut Metabolites from Gut Microbiome，encoder-decoder结构的模型，用于通过稀疏和可解释的潜在空间来捕获与炎症性肠道疾病相关的微生物-代谢物关系

+ 数据匮乏

  + 解决方法：数据增强，非监督学习，迁移学习，混合模型，**后两个还有待探索**

+ 模型评估

  