## Are Transformers Effective for Time Series Forecasting？

### TSF：time series forecasting 

+ 传统的统计方法：ARIMA

+ 机器学习方法：GBRT

+ 深度学习方法：Recurrent Neural Networks，Temporal Convolutional Networks 

### LTSF：long-term time series forecasting

+ LogTrans
+ Informer
+ Autoformer
+ Pyraformer
+ Triformer
+ FEDformer 



Transformer，是位置变换无影响的，主要就是句子中单词位置的变化对整体语义的影响不大。

分析时间数据的时候，数值本身的没有意义的，主要是对一组与时间相关的连续变化的数值的建模。



这篇文章提出了一个及其简单的单层线性模型：LTSF-Linear



### 基本公式

![image-20230301150850112](http://picture.jacobx.top/markdown/image-20230301150850112.png)

C个变量，L是回顾窗口的大小（look-back windows size）

![image-20230301151738283](http://picture.jacobx.top/markdown/image-20230301151738283.png)

第 i 个变量在第 t 个时间点的值



基本任务是预测未来 T 个时间点的值

![image-20230301151938760](http://picture.jacobx.top/markdown/image-20230301151938760.png)



有两种策略

+ IMS:  iterated multi-step，一次性预测一步，迭代得到多步的预测
+ DMS:  direct multi-step，一次直接优化多步预测目标

+ IMS采用的自回归估计，与DMS相比，方差较小，但是会受到误差累积的影响

+ 当存在高精度单步预测器且 T 相对较小时，IMS 预测更可取

+ 难以获得无偏单步预测模型或 T 较大时，DMS 预测会产生更准确的预测。



### 基于Transformer的LTSF解决方法

基于Transformer的关注点在较少探索的长期预测问题

局限性：

+ 原始自注意方案的二次时间/记忆复杂度
+ 自回归译码设计引起的误差累积



Informer，提出了一个新的transformer的架构和一个DMS预测策略减少了复杂读

更多Transformer变体将各种时间序列特性引入其模型，以提高性能或效率

![image-20230301162937968](http://picture.jacobx.top/markdown/image-20230301162937968.png)

#### 数据预处理阶段

+ 做 0 均值归一化。

+ Autoformer首先在每个神经块后面应用季节性趋势分解（seasonal-trend decomposition），这是时间序列分析中的标准方法，以使原始数据更具可预测性。
+ 具体来说，他们在输入序列上使用移动平均kernel（average kernel）来提取时间序列的趋势周期（ trend-cyclical）分量。原始序列和趋势分量之间的差异被视为季节分量（ seasonal component）
+ 在Autoformer分解方案的基础上，FEDformer进一步提出了混合专家策略，以混合不同核大小的移动平均核提取的趋势分量

#### 输入embedding的策略

+ self-attention 层里面是没有办法保存时间序列的位置信息的，但是时间序列的顺序是十分重要的
+ 目前的解决办法是embedding里面多加几层，加上固定位置编码、通道投影嵌入和可学习的时间嵌入层来保存时间序列的顺序信息
+ 加上时间卷积层，可学习的时间戳的时间embedding

#### Self-attention 策略

+ 主要是降低计算复杂度的工作

+ LogTrans和Pyraformer明确地将**稀疏性偏差引入到自我注意方案中**
+ LogTrans使用Logsparse掩码将计算复杂度降低到O（LlogL）
+ Pyraformer采用pyramic attention，以O（L）时间和内存复杂度捕获分层多尺度时间依赖。
+ Informer和FEDformer在自我注意矩阵中使用低秩属性。
+ Informer提出了ProbSparse自我注意机制和自我注意提取操作，以将复杂度降低到O（LlogL）
+ FEDformer设计了一个傅里叶增强块和一个随机选择的小波增强块，以获得O（L）复杂度。
+ Autoformer设计了一种串联自相关机制来取代原来的自我注意层。

#### Decoder层

+ Informer设计了一个生成式的decoder，来做DMS的预测
+ Pyraformer用一个全连接层来链接时空轴作为decoder
+ Autoformer将趋势周期分量的两个精细分解特征和季节分量的叠加自相关机制进行汇总，得到最终预测。
+ FEDformer使用分解方案和建议的频率注意块来解码最终结果。



Transformer 模型的前提是成对元素之间的语义相关性，而自注意力机制本身是排列不变的，其对时间关系建模的能力在很大程度上取决于与输入标记相关的位置编码。考虑到时间序列中的原始数值数据（例如，股票价格或电价），它们之间几乎没有任何逐点语义相关性。

在时间序列建模中，我们主要关注一组连续点之间的时间关系，这些元素的顺序而不是配对关系起着最关键的作用。

虽然采用位置编码和使用令牌嵌入子系列有助于保留一些排序信息，但排列不变自注意机制的性质不可避免地导致时间信息丢失。





### The simplest DMS model via a temporal linear layer

![image-20230301170830473](http://picture.jacobx.top/markdown/image-20230301170830473.png)

假设我们有一个10个变量的单序列，历史长度100，并假设预测未来50个时间步，则单个样本是一个100X10的time series matrix。

==10 X 100 -------》10 X 50==

这里的操作是，打平，变成 10 个 100维的向量，每个100维的向量直接接一个 100*50的linear layer，10个100维的向量得到10个50维的output，然后这些output 做求和即可。另外这里的linear layer的权重是share的，所以可以看到整个模型的结构是非常简单的，参数量很小了。



DLinear和Nlinear是当前liner的升级版

具体来说，DLinear是Autoformer和FEDformer中使用的分解方案与线性层的组合。它首先通过移动平均kernel和季节分量将输入的原始数据分解为趋势分量。然后，将两个一层线性层应用于每个组件，并将这两个特征进行汇总以获得最终预测。通过显式处理趋势，当数据中有明确的趋势时，DLinear增强了普通线性的性能。





现有的基于Transformer的模型在回望窗口增大时性能会恶化或保持不变。相比之下，所有LTSF Linear的性能都随着回望窗口大小的增加而显著提高。

因此，如果给定一个较长的序列，现有的解决方案往往会覆盖时间噪声，而不是提取时间信息，并且输入大小96完全适合大多数Transformer。