# tokenzier

tokenzier的目标就是把文字转化为模型能够处理的数字

###  word-based tokenizer

把句子拆分成词

给每一个词映射到一个ID，映射出来的词典是非常大的，然后没有办法识别词汇的时态变化和数量变化，会出现很多的UnKnown的情况。



### character-based tokenizer

把句子拆分成字母

+ 生存的词汇表实际上是更小的
+ Unknown token的词汇数量也变少了，应为每一个词都是有字母构建的
+ 但是由于拆分的过于零散，减少了词汇本身的意义
+ 生成了过多的token，基于词的生成一个token，基于字母的生成好多token



### subword tokenization

把词汇拆分成子词，拆分到不能继续得到字词为止

+ 能够得到词汇本身的语义
+ 能够节省存储空间，使用较少的token得到较高的覆盖率





### More

+ GTP-2，使用的是Byte-level BPE
+ BERT，使用的是WorldPiece
+ 一些跨语言的模型，使用的是SentencePiece 或者 Unigram





### 加载和保存Tokenizer

实际上和加载或者保存模型是一样的

```python
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

# AutoTokenizer会更具checkpoint的名字来加载相应的 tokenizer
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
```



### Encoding

分为两步，第一步是tokenization ,第二步是把token转化为IDs



tokenize()  方法得到token的表示

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

sequence = "Using a Transformer network is simple"
tokens = tokenizer.tokenize(sequence)

print(tokens)

tokenization 得到的是分词的token
['Using', 'a', 'transform', '##er', 'network', 'is', 'simple']
```



convert_tokens_to_ids()，把token转化为数字，decode()方法在生成式任务，Seq-to-Seq的任务上有很好的应用

```python
ids = tokenizer.convert_tokens_to_ids(tokens)
print(ids)

[7993, 170, 11303, 1200, 2443, 1110, 3014]
```



### Decoding

decode()方法来解码ids

```python 
decoded_string = tokenizer.decode([7993, 170, 11303, 1200, 2443, 1110, 3014])
print(decoded_string)

'Using a Transformer network is simple'
```



