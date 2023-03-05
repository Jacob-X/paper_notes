Byte-Pair Encoding (BPE) 最初是一个用来压缩文本的算法，是openai用来对GTP进行tokenization的算法

> Byte-Pair Encoding (BPE) :  GPT, GPT-2, RoBERTa, BART, DeBERTa



```
"hug", "pug", "pun", "bun", "hugs"
```

 学习的词汇表就会从`["b", "g", "h", "n", "p", "s", "u"]`这几个字母开始，对于从来都没有遇到过的特殊字符，会转化为unkwon tokens,对于从来都没有见到过的字符，NLP的结果往往都不会很好

> The GPT-2 and RoBERTa tokenizers 使用  *byte-level BPE*  ，包含了所有组成字符的字节（bytes）而不是Unicode，减少了unknown的数量 

学习融合两个元素的规则形成词汇，BPE算法将搜索最频繁的token pairs, token pairs就是需要融合的内容



假如有如下的频率

```python
("h" "u" "g", 10), ("p" "u" "g", 5), ("p" "u" "n", 12), ("b" "u" "n", 4), ("h" "u" "g" "s", 5)
```

("u", "g")是出现频率最高的pair,就把这两个token融合起来



融合之后就得到了如下的结果

```python
Vocabulary: ["b", "g", "h", "n", "p", "s", "u", "ug"]
Corpus: ("h" "ug", 10), ("p" "ug", 5), ("p" "u" "n", 12), ("b" "u" "n", 4), ("h" "ug" "s", 5)
```



再进行融合

```python
Vocabulary: ["b", "g", "h", "n", "p", "s", "u", "ug", "un"]
Corpus: ("h" "ug", 10), ("p" "ug", 5), ("p" "un", 12), ("b" "un", 4), ("h" "ug" "s", 5)
```

