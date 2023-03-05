# Unigram tokenization

使用Unigram tokenization的模型：

+ AlBERT
+ T5
+ mBART
+ Big Bird
+ XLNet



## Training algorithm

从一个大的词汇表开始，不断的从里面移除token，直到得到一个想要的词汇表的大小

构建基本词汇表的方法：

1. 从 pre-tokenization 的词汇表里选出常用的子词
2. 在 large size vocabulary 的语料库上用BPE初始化一下



在训练的每一步， Unigram algorithm都会计算当前的词汇表在语料库的loss

对于词汇表中的每个符号，该算法计算如果删除该符号，整体损失会增加多少，并寻找增加最少的符号

这个训练过程是非常昂贵的，一般将损失的比例（作为一个超参数）定义在10-20

有一个前提：永远不移除基本的character，保证所有的单词都可以tokenized





## Tokenization algorithm

1. 首先，假定每个token都是独立的，单个token的概率只与其本身有关

2. 每个token的概率实际上是在语料库中出现的频率

3. 对word的tokenization的概率计算，是将每个字母在语料库中的概率累乘

   example: “pug” 组成token的概率

   ![image-20230305183713448](http://picture.jacobx.top/markdown/image-20230305183713448.png)

4. 找到所有的可能的segmentations，然后计算他们的概率，这里可以使用*Viterbi algorithm*（维特比算法），计算出最有可能的子词



## Implementing Unigram

```python
corpus = [
    "This is the Hugging Face Course.",
    "This chapter is about tokenization.",
    "This section shows several tokenizer algorithms.",
    "Hopefully, you will be able to understand how they are trained and generate tokens.",
]
```



```python
from transformers import AutoTokenizer
#使用xlnet作为基本的模型
tokenizer = AutoTokenizer.from_pretrained("xlnet-base-cased")

from collections import defaultdict

word_freqs = defaultdict(int)
for text in corpus:
    words_with_offsets = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
    new_words = [word for word, offset in words_with_offsets]
    #下面这个for循环统计了每个词的频率
    for word in new_words:
        word_freqs[word] += 1

word_freqs
```



```python
char_freqs = defaultdict(int)
subwords_freqs = defaultdict(int)
for word, freq in word_freqs.items():
    for i in range(len(word)):
        char_freqs[word[i]] += freq
        # Loop through the subwords of length at least 2
        for j in range(i + 2, len(word) + 1):
            subwords_freqs[word[i:j]] += freq

# Sort subwords by frequency
sorted_subwords = sorted(subwords_freqs.items(), key=lambda x: x[1], reverse=True)
sorted_subwords[:10]
[('▁t', 7), ('is', 5), ('er', 5), ('▁a', 5), ('▁to', 4), ('to', 4), ('en', 4), ('▁T', 3), ('▁Th', 3), ('▁Thi', 3)]
token_freqs = list(char_freqs.items()) + sorted_subwords[: 300 - len(char_freqs)]
token_freqs = {token: freq for token, freq in token_freqs}
```



把频率转化为概率

```python
from math import log

total_sum = sum([freq for token, freq in token_freqs.items()])
model = {token: -log(freq / total_sum) for token, freq in token_freqs.items()}
```



使用维特比算法对单词进行标记

+ 该算法计算单词每个子串的最佳分割，我们将其存储在名为 best_segmentations 的变量中。

+ 我们将为单词中的每个位置（从 0 到其总长度）存储一个字典，其中包含两个键：最佳分割中最后一个标记的开始索引和最佳分割的分数。 

+ 使用最后一个标记开始的索引，一旦列表完全填充，我们将能够检索完整的分段。

+ 填充列表只需要两个循环：主循环遍历每个起始位置，第二个循环尝试从该起始位置开始的所有子字符串。 

+ 如果子字符串在词汇表中，我们将对该词进行新的分割，直到该结束位置，我们将其与 best_segmentations 中的内容进行比较。 
+ 一旦主循环完成，我们就从末尾开始，从一个开始位置跳到下一个开始位置，边走边记录记号，直到我们到达单词的开头

```python
def encode_word(word, model):
    best_segmentations = [{"start": 0, "score": 1}] + [
        {"start": None, "score": None} for _ in range(len(word))
    ]
    for start_idx in range(len(word)):
        # This should be properly filled by the previous steps of the loop
        best_score_at_start = best_segmentations[start_idx]["score"]
        for end_idx in range(start_idx + 1, len(word) + 1):
            token = word[start_idx:end_idx]
            if token in model and best_score_at_start is not None:
                score = model[token] + best_score_at_start
                # If we have found a better segmentation ending at end_idx, we update
                if (
                    best_segmentations[end_idx]["score"] is None
                    or best_segmentations[end_idx]["score"] > score
                ):
                    best_segmentations[end_idx] = {"start": start_idx, "score": score}

    segmentation = best_segmentations[-1]
    if segmentation["score"] is None:
        # We did not find a tokenization of the word -> unknown
        return ["<unk>"], None

    score = segmentation["score"]
    start = segmentation["start"]
    end = len(word)
    tokens = []
    while start != 0:
        tokens.insert(0, word[start:end])
        next_start = best_segmentations[start]["start"]
        end = start
        start = next_start
    tokens.insert(0, word[start:end])
    return tokens, score
```



```python
print(encode_word("Hopefully", model))
print(encode_word("This", model))

(['H', 'o', 'p', 'e', 'f', 'u', 'll', 'y'], 41.5157494601402)
(['This'], 6.288267030694535)
```



计算模型在语料库上的loss

```python
def compute_loss(model):
    loss = 0
    for word, freq in word_freqs.items():
        _, word_loss = encode_word(word, model)
        loss += freq * word_loss
    return loss

compute_loss(model)
413.10377642940875
```



计算删除每一个token之后，模型的loss

```python
import copy


def compute_scores(model):
    scores = {}
    model_loss = compute_loss(model)
    for token, score in model.items():
        # We always keep tokens of length 1
        if len(token) == 1:
            continue
        model_without_token = copy.deepcopy(model)
        _ = model_without_token.pop(token)
        scores[token] = compute_loss(model_without_token) - model_loss
    return scores

scores = compute_scores(model)

#由于“ll”用于“Hopefully”的标记化，删除它可能会使我们使用标记“l”两次，我们预计它会有正损失。 “his”仅在单词“This”中使用，它被标记为自身，因此我们期望它具有零损失。
print(scores["ll"])
print(scores["his"])
6.376412403623874
0.0
```



接下来要做的就是把special token加进来，然后循环直到我们从词汇表中删除了足够多的标记以达到我们想要的大小

```python
percent_to_remove = 0.1
while len(model) > 100:
    scores = compute_scores(model)
    sorted_scores = sorted(scores.items(), key=lambda x: x[1])
    # Remove percent_to_remove tokens with the lowest scores.
    for i in range(int(len(model) * percent_to_remove)):
        _ = token_freqs.pop(sorted_scores[i][0])

    total_sum = sum([freq for token, freq in token_freqs.items()])
    model = {token: -log(freq / total_sum) for token, freq in token_freqs.items()}
```



为了对一些text做token， apply the pre-tokenization and then use our `encode_word()` function

```python
def tokenize(text, model):
    words_with_offsets = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
    pre_tokenized_text = [word for word, offset in words_with_offsets]
    encoded_words = [encode_word(word, model)[0] for word in pre_tokenized_text]
    return sum(encoded_words, [])


tokenize("This is the Hugging Face course.", model)
['▁This', '▁is', '▁the', '▁Hugging', '▁Face', '▁', 'c', 'ou', 'r', 's', 'e', '.']
```

