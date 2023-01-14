# 大模型在生物医学的交叉应用

医疗对话，生物文本的特殊处理

## 生医自然语言处理

生医NLP的背景：

1. 分析生物文献、药物说明书、医疗记录、实验指导书
2. 药物分析、药物研发、元分析、辅助诊断
3. 对领域特殊数据：蛋白质、DNA、有机分子结构，效果显著



特点：

1. 海量未标注的数据，PUBMED
2. 通常使用弱监督、无监督模型
3. 有很高的知识门槛，在架构中需要考虑这样的问题，Konwledge-echanced learning



生医文本挖掘，从非结构化的文本中中找到结构化的生医专业知识

1. 实体信息挖掘，各种生医名词NER，
2. 命名实体链接NEN，将识别出来的实体链接到知识库里的条目，利用知识库的信息



1. 主题识别，用有监督的模型 topic recognition
2. 给每一篇文献标识知识库中的术语，多标签的识别任务，索引indexing



1. 关系抽取，抽取实体之间的关系
2. 事件抽取，在实验指导书、说明书中常使用，在什么情况下的步骤，具体的操作



1. pathway、hypothesis
2. Pathway extraction: pathway特指生物医学内物质之间的相互作用，关乎基因表达、蛋白质的合成、生理复杂的变化，对于研究疾病的致病机理非常重要，一般使用关系抽取、事件抽取的方法，规则匹配
3. literature-base discovery：根据线索，发现实体之间的关系，综合已有的信息进行推断，发现隐藏的新知识，基于ABC co-occurence model based，疾病A由B物质引发，药物C可以降低B物质的含量，推断C对A有治疗效果。发现潜在关联和新的假说，找到新的研究方向。



从非结构化的数据中找到结构化的生医专业知识

PipLine:

1. 先做命名实体识别，找到文本中的实体`NER`
2. 命名实体链接，将实体连接到知识库中`NEN`，命名实体规范化（Named entity normalization，NEN）也叫实体链接（entity linking）
3. 关系抽取，抽取实体之间的关系`RE`



NER：CNN, BiLSTM+CRF, 随着预训练模型的出现，现在常使用BERT+CRF, BERT+Prompt来作为模型的主干网络（backbone）



NEN: representation + distance

使用一些通常的方法，预训练词向量，预训练模型、图神经网络，将文本中实体的mention嵌入到一个包含上下文和专业知识信息的representation中，将知识库中的KB信息也嵌入到一个包含实体名和专业知识的representation，最后使用距离的方法来进行匹配 

在生物医学领域，实体间的重名和缩写是非常常见的，NER抽取出来的实体名可能与知识库中的很多实体的名称或缩写重复，必须使用实体的上下文信息和KB中的专业信息才能找到准确的结果

NEN的关键点：实体名称的去模糊化，具体的操作方式是：实体上下文理解+ 知识库的信息



生医NER, NEN 常用工具，

ScispaCy: 适用于各种生医的语义分析问题（句子分割，依赖分析）也可以用来构建远程标签

Pubtator: 基于PubMed+PMC，进行实体分析和实体链接的操作

开放领域的实体挖掘待进一步研究

近年常用:

1. BERT + BILSTM + CRF, 算是最长用的NER架构方法, 一般使用scibert,biobert的预训练模型，BILSTM利用长短时记忆的特点来学习一定距离的依赖关系（但是可能不能明显提升实验的结果），CRF对序列标签进行了统计建模，最后用一个线性层来推断某个位置是实体还是非实体

2. BERT + Prompt 

   进行一个命名实体分类的任务，使用prompt来构建模板，使原来的分类任务变成一个mask任务，缩小下游任务与预训练任务的表现差距





关系抽取：

RE分为两类：sentence level, document level

sentence level: 两个实体只出现在同一个句子的情况，常用的benchmark: ==ChemProt==（句子级别的化学物质和蛋白质之间的关系，将常见关系分类为10组，是多分类的任务），==PPI==（抽取句子级别蛋白质和蛋白质之间的相互作用）

document level： 根据文章的多个句子进行推断，常见的benchmark：==BC5CDR==(抽取摘要级别的化学物质和疾病的关系)，==GDA==(抽取基因和疾病的关系)

常见的方法是

1. 基于预训练模型，以一段文字作为输入，获取到data的表示之后再进行关系的分类

2. 基于图和图神经网络的方法，将实体和关系建模为图网络的点和边，生医领域可以用来关系抽取的数据比较少，对关系间的类别也缺少系统间的定义，现在的很多工作只进行了有无关系的判断（2分类）

   ==关系抽取任务会向多分类方向进行==



文档级关系抽取的常用模型：

1. 基于预训练模型的机械模型，为了给PML提示实体的位置和类型，会在实体的两端增加一些mark, 标识mention的开始和结束，这里的mention可以是多个单词token，输入的文本进行一个预训练语言的编码，取每个mention的开始的隐状态作为每个mention的表示，然后做一个最大池化，得到每个实体的表示，最后经过一个双线性层和分类器，得到关系的分类和类别

   缺点：不能解决文档级抽取的一些问题，多跳推断：两个实体关系的推出需要一个中间实体，不能直接从文档中直接得到

2. 用图神经网络模型，如果两个实体在同一个句子中出现，就用边连接起来，节点和边的初始序列表示都可以从预训练模型的编码的隐状态序列得到，随后通过卷积神经网络的迭代更新就可以得到节点和边的表示，在第一轮迭代完成后，将部分图去修改成了一个完全图，在计算邻接矩阵的时候，通过多头注意力的机制，融合所有节点之间两两交互的信息，使得信息流可以在各节点充分的流动，从而有助于多跳推断的进行。



文本挖掘存在的问题：

生医领域的文本标注的成本非常昂贵，ChemProt: chemical - proteins, 1820条标注数据，BC5CDR：chemical - disease,1500条数据 

解决的方法是; 

1. 使用预训练的模型来训练大量未被标注的数据
2. 使用一些标注工具来对数据进行粗标注，进行弱监督的训练，这是远程监督的做法

解决案例：

+ 收集到PUBMED中数据，使用NER+NEN来进行关系的抽取和连接，连接到CTD（一个知识库）的实体，再引入远程相关的假设，在ctd中的两个实体，如果恰好出现在文章的同一个句子中，则句子就表达了AB的关系，这个假设就可以创建高效的训练样本，
+ 但是远程监督的假设大部分都是不成立的，实际上斌没有表达二者之间的关系，会产生许多假阳性的噪声，CTD没有包含所有的实体关系，因此会造成假阴性的噪声
+ 远程监督的场景下进行降噪，自监督降噪，多实例学习，预训练降噪都是常用的方法
+ 预训练降噪：可以先使用一些人工标注的数据来先训练一个预降噪的模型，然后使用这个模型来对远程监督的模型来进行一个打分，每次只保留分数最高的一部分作为远程监督的训练样本，可有效的降低噪声比例
+ 多实例学习：将若干个同一实体对的样例打包成一个bag，在bag层面使用attention来选择执行度得分较高的样例，attention可以动态分配权重，筛除噪声的同时可以避免一刀切的影响，避免信息的过度缺失
+ 自监督学习：
  1. 在第一阶段正常的使用远程监督的方法来训练模型，同时通过控制模型训练的轮数和步数，引入early-stopping，防止模型对噪声数据产生一个过拟合的结果  
  2. 采用self-training的方法，引入一个teacher模型和student模型，teacher会使用第一阶段（初步具备NER能力）的权重来进行一个初始化
  3. 然后采用student模型，student不在拟合远程监督的原始标签，而去拟合由teacher产生的一个伪标签，在student每轮训练结束之后，再去更新teacher的权重，将teacher的权重更新到与student一致，在更新的过程中，伪标签的噪声比例也会越来越低，从而达到逐渐降低噪声的效果





生医领域的PLM:

1. 专业资料训练：Sci-BERT, BioBERT, clinical BERT
2. 引入特殊任务（NER、NEN）：KeBioLM（NER+entity linking）、MC-BERT(entity/pharse masking + representation )



以PLM为主干网络，做完一连串的pipline之后，就得到了结构化的知识，并且可以组织为（KB（知识库）或者KG（知识图谱）），KB和KG的构建是生医文本挖掘的重要应用，实现了非结构化到结构化知识的转化

比较常用的KB:

1. Mesh: 涵盖了疾病和大量的化学物质
2. UMLS: 包含了各种各样的生医术语
3. NCIBI Gene
4. UniProt



KG需要关系抽取，所以现存的KG数量比较少

常用的KG:

1. CTD: 类别较全
2. DisGeNet: 疾病和基因的关系
3. HuRI: 疾病和蛋白的关系

知识图谱存在不完整性的缺陷



模型可以通过使用KG和KB来提高模型在下游任务上的表现

在解决实体任务的消息时，结合实体mention的上下文一级UMLS知识库中的实体名，类别和描述信息



将文本挖掘的知识注入到PLM中方法：adapters, Customized pretraining tasks Prompt tuning, Delta Tuning



SMedBert (Enhanced PLM)

不仅引入了mention的实体信息，还会从知识库中找出于该链接实体K个相邻的实体，找到相邻实体与链接实体的关系



### 生医文本挖掘技术的应用

1. NER和NEN能提高我们获得实体信息的效率

2. 建立了文本和知识库的桥梁
3. 将口语的表达转化为标准的科学术语
4. QA assistance,AI问答



知识图谱难以更新，并且不能覆盖文献里所有的信息，间接模糊的关系也没有被KG所包含，所以要去文献库里进行查找

1. 现有的生医搜索引擎，需要关系抽取的功能

   用二分类来处理搜索引擎召回的文章，计算置信度得分，同时使用对比学习的方法来获得置信度得分最高的结果，置信度越高，文章中对应的两个实体的关联度可能就越高，使用文档级关系抽取的模型，我们可以使模型具有跨界推断的能力，可以补全KG中模糊和缺失的部分信息，同时也帮助生物的研究者提高了获得信息的效率

2. 在生物医学研究的临床分析中，对实验记录进行分析

   分析不同condition下的不同treatment带来的不同的result，进行一个结果分析，获得多学术和治疗有帮助的结论



## Diagnosis Assistance

辅助诊疗是面对普通大众人群的应用，自主医疗系统

文本挖掘是面对生医领域的专家或者研究者进行知识筛选与抽取的过程

Diagnosis Assistance分为两个大的任务：

1. ==Text Classification==：automatic triage(自动分诊) & medicine prescription （药物处方）

   

   相关数据集：

   + MDG(中文数据集，医患对话标注实体，症状分类)
   + MIE: 症状，检查项目，手术，疾病，药物的分类

   Backbones: SVM, LSTM, Bert, GPT



​		MIE提供了一个单独的技术路径，医患对话的时候会对文本进行编码(encoder)，对所有的分类候选项也做一个编码，通过一个匹配模块来进行match操作，最后得到一个类型分布的分数。==现有的一个提升思路，对分类的candidates进行一个知识注入==



2. ==Dialogue==

   datasets : MedDialog(大规模的中文数据集，医患对话和最终结论)

   对话是典型的文本生成任务，通常是多轮且没有可选参考答案的，灵活性高

   流程：

   ![image-20221026110433708](http://picture.jacobx.top/markdown/image-20221026110433708.png)

​		

对话系统这里是task-oriented的任务，就是正对某个领域进行特定的任务，需要连接相应的知识库系统

比较常见的实现方法是==retrieval-based(检索式)==的对话系统

会有一个特别大的对话库，在输入新的文本的时候会进行检索，找是否存在相似的对话，找到相似的对话就能得到当时的回复，调用出来来回复患者。

现在就是把PLM和retrieval-based结合起来，分类，还是去对话库中进行检索，然后找到相似的结果后进行文本生成来得到最终的回复。

retrieval-guide，retrieval系统可能达到的答案是不相关的，所以训练一个匹配的模块，然后比较retrieval结果和自己生成的结果，给retrieval打分。



knowledge-based dialogue  

实际上，实现的时候是在大的知识图里面找一个知识子图，可以使用Graph-transformer来把拓扑图进行编码，和输入的原始文本的编码一起来影响注意力层，从而进行任务的下游生成



medical dialogue

需要保证安全性（对于从大量数据中检索到的知识，不能乱答，然后生成的结果要有可解释性），医疗知识的交互过程，对于患者提出的医疗问题，要能生成人类可理解的解决方案，让患者能够理解

模型对于用户：要能提取出经验知识

用户对于模型：要询问存在的知识问题，模型要取出用户想要的知识



medical dialogue的特点：

1. 患者口语化的语言和知识库里结构化的标准的语言是存在gap的
2. entity linking（一个物体的俗名、标准名、重名、缩写） and standarlization（对于可解释性和知识支撑非常重要的过程） 对于对话系统的重要性
3. 患者的隐私保护问题



entity linking可以看作一个标准的信息检索，最开始有一个粗排，然后有一堆的候选term，然后通过一些算法（BM25）来筛选出与检索内容相关的术语，然后用Bert来做精排，打分，找到与当前的mention最匹配的concept

比较常用的手段：triplet network, 使用同一个模型来输出mention和concept的表征，使用同一个模型来训练一个正例和一个负例，做一个self-learning,这样的训练方式，可以让下游任务的结果得到很好的提升。例子：pubmedBert,一个使用pubmed来预训练的模型，通过triplet network的训练方式，可以很好的将同类项（意思相同表达不同）聚集的分布在一起



KB enhancing：

做了术语的标准化之后，在知识库中对实体进行检索，找到与实体有联系的实体，然后将原始文本和检索得到的实体一起输入到Transformer decoder中，这种==原始文本＋检索实体==的方法可以巨幅的增加下游任务的效果

![image-20221026211319988](http://picture.jacobx.top/markdown/image-20221026211319988.png)



Patient states & Physician policies

state，患者提到症状

Physician policies，诊疗结果和药物建议

模型网络结构：1.对一句话得到一个state tracker，2.把这句话的原始文本和state 一起放到action的模块中进行训练

这样的模型就不是黑箱的模型了



清华的探索：

1. 在预训练的过程中就埋下一些prompt，对医疗对话的每一个任务都有一个单独的soft prompt,让语言模型提前适应下游任务
2. 将医疗对话的文物分成两阶段的任务，会对下游任务有提升，首先生成一个topic，然后用topic的概念再从知识库中取到一些三元组（疾病对应的症状和用药），然后把三元组和原始文本拼到一起来指导下游任务的训练



## Substance Representation

处理生物的表征：化学物质，小分子，蛋白质，DNA

==自然语言处理可以处理线性的文本==，只要生物的信息可以被处理成线性的，那就可以将自然语言处理应用在生物信息上





### DNA

研究的主要是DNA的非编码区，编码区是索然无味的，非编码区有非常丰富的基因表达和调控的基因

非编码的任务：

1. 预测基因的表达
2. 预测启动子和转录因子结合位点

非编码区的序列较长，4000-5000个碱基，普通的PLM,例如BERT的最长的token = 512，所以长度会有显著的差异



datasets:

1. 人类基因组，CRCh38/hg38
2. Cap Analysis Gene Expression (CAGE)
3. Descartes, Human Chromatin Accessibility During Development



自然语言处理的模型能有效的从大量的数据中获取特征和潜在规律

Transformer是适合长距离的依赖关系的，attention

DNABert模型架构：对很长的序列首先过一层CNN,找到一些对于下游任务来说重要的区域，Bert本身无法处理过长的序列，但是这个操作会对后续模型的加速造成影响和阻碍



DNA是有四个碱基构成的，在word embedding的时候，生成的token就会少很多（对于正常自然语言处理的任务来说）

+ 对于DNA来说，碱基的组成序列是非常重要的，==碱基的排列位置很重要==
+ 对DNA的任务来说，要对输入的方式进行一定的调整，==使用K-mer这种滑窗的方式来进行输入==，可以增加下游的信息含量，提高下游任务的表现



### Protein

主要关注于氨基酸序列

任务：

1. 蛋白质有四级结构，是比较难用线性文本来进行表示的，暴力的解法（三维空间坐标），通常我们认为一个氨基酸序列其实是与最后的空间结构是由一个强对应关系的，所以==模型的一个重要的任务就是预测蛋白质的空间结构==
2. 预测同源进化关系
3. 蛋白质工程，例如转接荧光蛋白、提升蛋白质的稳定性等任务



datasets:

1. Uniref
2. GO annotations，有精细标注
3. Protein Data Bank



蛋白质序列常用的模型：

1. BiLSTM + CRFS
2. Autoencoder models
3. big model: ![image-20221026225056400](http://picture.jacobx.top/markdown/image-20221026225056400.png)

​	蛋白质的预训练模型，暂时这些任务上的难度不需要过大的参数，预训练的模型有很强的迁移性

4. Alpha-Fold, 最振奋人心的研究成果，给一个DNA序列，可以预测出蛋白质的3D结构，总体来说融合了很多的信息，MSA+EvoFormer+End2end

   一代的时候都是独立的模块，二代的时候是将几个模块都串联在了一起形成了端到端的神经网络

5. MSA Transformer, 是受到Alpha-Fold的启发，是对同源序列进行分析，横向attention分析，分析两个碱基是否有可能结合，纵向是分析有无突变的发生



Alpha-Fold模型结构：

1. EvoFormer：在氨基酸序列处理上的Transformer,这个模块重复了48次，巨大的模型，但是已知的氨基酸序列对应的蛋白质结构的标注数据是非常少的，所以是不够用的，所以在进行training之前，首先对许多的序列进行了mask learning, 做了一个无监督pre-training的过程，然后再去初始化EvoFormer模块
2. 就是是做了pre-training之后，标注的数据还是显得非常少，因此首先使用标注的数据粗糙的训练一个模型参数（不要过拟合），浅训之后，再对剩下的无处理的数据进行预测，打上标注
3. 最后的时候是将精标的数据和用模型预测标注的数据放在模型中一起训练的



具体的网络结构：

1. EvoFormer的模块：MSA row/column attention templates, 残基对的表示，并不是对序列中的每一个序列进行标识，而是拉成一个二维的表，两两碱基之间一个对会有一个representation（Pair-wise representation）,representation结合MSA得到最后的表征
2. Pair-wise representation graph iterative update,
3. EvoFormer得到的single representation，通过图网络分析得到的pair-wise representation, 使用blackhole initialize生成肽键的键角和距离，最终得到蛋白质的结构





### Chemicals

Molecular fingerprints

Molecular graphs  -> GCNs

SMILES strings -> LMs

任务：

1. molecule property classification，分子成分的分类
2. chemical reaction classification，化学反应的分类，做化学物质的有机合成，知道材料和产物，但是中间路径是未知的，reaction的分类对提取template是很重要的一件事



Dataset:

1. MoleculeNet
2. USPTO 1k TPL





使用bert的时候需要对token进行特定的设定，因为原始的token是针对英文的，需要设置特定的官能团，使其不要切碎，可视化的结果可以对分析某个小分子带来启发



清华实验室的一个成果：KV-PLM

把chemical转化为线性文本，然后用自然语言的模型去处理，化学物质的结构和知识库中的信息实际上是互补的，分子结构式是非常精确、直白的，文本的描述是更加灵活多变的，可读性更强

那就可以把两种的信息同时利用起来，文本中可以提到化学物质的性能，有一个mapping correlation的过程，实际上就是一个多模态的处理，用同一个模型来处理化学物质的结构和文本的描述，

过程：

先对化学的结构式进行tokenlized 的预处理，然后在文本描述的部分将分子的结构式拼到文本的描述中，这里拼接的是smiles strings ，最后放到Bert的一个backbone中，最后mask的时候会遮掉一些部分，模型可以自动地学习到官能团的一些潜在性质，学习到化学结构和文本描述的关系，找出化学结构的子结构的功能？例如：找到影响分子溶解性的官能团

下游任务：

1. 让模型进行chemical exam, property prediction(给一个分子结构，给四个性质预测的选项，进行选择)
2. 药物发现 drug discovery, 输入文本描述，找出符合要求的分子，**问题在于表达式的空间表达式无法表达，后续的主要研究方向**



现在需要做的事情：搭建一个好的benchmark，一个高质量的benchmark对推动一个领域的发展是十分重要的



Biomedical NLP 当前可能存在的一些Future Directions :

1. ==konwledgeable big model==: models with more expert knowledge achieving better performance，生医是一个动态发展的，知识属性很强的学科，模型必须具备足够多的expert knowledge才能实现更好的表现
2. ==AI for science==, user-friendly assistant tools with lower barriers to entry ; unleash human researcher productivity，从业者对一些辅助工具的使用率不是特别高，工具的使用门槛太高了，需要使用友好的辅助工具
3. ==cross-modal processing==, bridging vision & language information or different forms of data, 做跨模态的处理，融合处理不同的数据类型，后续的医疗检查结果图像的诊断
4. ==low-resource learning :== lack of annotated data , 对于生医领域，数据标注的成本太高了，导致了标注的数据特别少，pre-training model 就是来解决数据不够的问题



