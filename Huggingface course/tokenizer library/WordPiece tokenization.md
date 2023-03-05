# WordPiece tokenization

 使用的模型：BERT, such as DistilBERT, MobileBERT, Funnel Transformers, MPNET

Google没有开源，接下来的内容是基于已发布的文献复现的

把一个词分割成一些子词，然后在子词之前加上前缀##

和BPE不同的是，BPE是根据token pairs出现的频率来进行融合规则的学习，WordPiece 是 对每个pair进行一个分数的计算

> score=(freq_of_pair)/(freq_of_first_element×freq_of_second_element)

优先合并各个部分在词汇表中出现频率较低的pair

```python 
("hug", 10), ("pug", 5), ("pun", 12), ("bun", 4), ("hugs", 5)


first merge learned is ("##g", "##s") -> ("##gs").

Vocabulary: ["b", "h", "p", "##g", "##n", "##s", "##u", "##gs"]
Corpus: ("h" "##u" "##g", 10), ("p" "##u" "##g", 5), ("p" "##u" "##n", 12), ("b" "##u" "##n", 4), ("h" "##u" "##gs", 5)
```

