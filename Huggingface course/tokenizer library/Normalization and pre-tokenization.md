![The tokenization pipeline.](http://picture.jacobx.top/markdown/tokenization_pipeline.svg)



#### normalization数据标准化

+ 移除不需要的空格
+ 字母都变成小写
+ removing accents



Transformers `tokenizer` 有一个变量 backend_tokenizer 提供了tokenizer library中存在的tokenizer

==tokenizer.backend_tokenizer.normalizer.normalize_str==，数据标准化的方法

```python 
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

print(type(tokenizer.backend_tokenizer))
<class 'tokenizers.Tokenizer'>

print(tokenizer.backend_tokenizer.normalizer.normalize_str("Héllò hôw are ü?"))
'hello how are u?'
```



#### pre-tokenization

把句子分割成较小的实体

==backend_tokenizer.pre_tokenizer.pre_tokenize_str==

```python
tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str("Hello, how are  you?")

[('Hello', (0, 5)), (',', (5, 6)), ('how', (7, 10)), ('are', (11, 14)), ('you', (16, 19)), ('?', (19, 20))]
```

每一个模型的tokenizer 的算法不同，分割出来的结果也就相应的不同，BERT,GTP,T5(SentencePiece algorithm)



#### SentencePiece algorithm

1. 将句子考虑为Unicode characters的序列
2. 用特殊的字符( _ )来替代空格 
3. 与Unigram algorithm结合使用，就不需要pre-tokenization的过程了，对不适用空格的语言（中文，日文）效果好
4. 可以逆tokenization操作，把 _ 都换成空格就行， BERT的tokenization 的算法就不可逆，因为把连续的空格都替换成了一个空格了



#### Algorithm overview

+ BPE (used by GPT-2 and others)

+ WordPiece (used for example by BERT)

+ Unigram (used by T5 and others)

| Model         | BPE                                                          | WordPiece                                                    | Unigram                                                      |
| ------------- | :----------------------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Training      | 从小的词汇开始，学习融合tokens的规则                         | 从小的词汇开始，学习融合tokens的规则                         | 从大的词汇开始，学习删除tokens的规则                         |
| Training step | Merges the tokens corresponding to the most common pair      | Merges the tokens corresponding to the pair with the best score based on the frequency of the pair, privileging pairs where each individual token is less frequent | Removes all the tokens in the vocabulary that will minimize the loss computed on the whole corpus |
| Learns        | Merge rules and a vocabulary                                 | Just a vocabulary                                            | A vocabulary with a score for each token                     |
| Encoding      | Splits a word into characters and applies the merges learned during training | Finds the longest subword starting from the beginning that is in the vocabulary, then does the same for the rest of the word | Finds the most likely split into tokens, using the scores learned during training |
