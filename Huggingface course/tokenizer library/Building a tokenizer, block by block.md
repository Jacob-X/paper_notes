# Building a tokenizer, block by block



tokenization comprises several steps：

- Normalization (any cleanup of the text that is deemed necessary, such as removing spaces or accents, Unicode normalization, etc.)
- Pre-tokenization (splitting the input into words)
- Running the input through the model (using the pre-tokenized words to produce a sequence of tokens)
- Post-processing (adding the special tokens of the tokenizer, generating the attention mask and token type IDs)

![The tokenization pipeline.](http://picture.jacobx.top/markdown/tokenization_pipeline.svg)



### 从头建立一个tokenizer（build a tokenizer from scratch）

- `normalizers` contains all the possible types of `Normalizer` you can use (complete list [here](https://huggingface.co/docs/tokenizers/python/latest/api/reference.html#module-tokenizers.normalizers)).
- `pre_tokenizers` contains all the possible types of `PreTokenizer` you can use (complete list [here](https://huggingface.co/docs/tokenizers/python/latest/api/reference.html#module-tokenizers.pre_tokenizers)).
- `models` contains the various types of `Model` you can use, like `BPE`, `WordPiece`, and `Unigram` (complete list [here](https://huggingface.co/docs/tokenizers/python/latest/api/reference.html#module-tokenizers.models)).
- `trainers` contains all the different types of `Trainer` you can use to train your model on a corpus (one per type of model; complete list [here](https://huggingface.co/docs/tokenizers/python/latest/api/reference.html#module-tokenizers.trainers)).
- `post_processors` contains the various types of `PostProcessor` you can use (complete list [here](https://huggingface.co/docs/tokenizers/python/latest/api/reference.html#module-tokenizers.processors)).
- `decoders` contains the various types of `Decoder` you can use to decode the outputs of tokenization (complete list [here](https://huggingface.co/docs/tokenizers/python/latest/components.html#decoders)).



## Acquiring a corpus

使用 [WikiText-2](https://huggingface.co/datasets/wikitext) dataset

```python
from datasets import load_dataset

dataset = load_dataset("wikitext", name="wikitext-2-raw-v1", split="train")

# 生成一个1000条数据的batch
def get_training_corpus():
    for i in range(0, len(dataset), 1000):
        yield dataset[i : i + 1000]["text"]
        
```



tokenizer也可以直接在文本上进行训练

```python
with open("wikitext-2.txt", "w", encoding="utf-8") as f:
    for i in range(len(dataset)):
        f.write(dataset[i]["text"] + "\n")
```





## Building a WordPiece tokenizer from scratch

训练一个WordPiece tokenizer 

需要明确 `unk_token`，让模型遇到未知的character的时候知道返回什么，设置max_input_chars_per_word，word超过设置的长度就会被切分

```python
from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)

tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
```



BertNormalizer

+ lowercase 
+ strip_accents
+ clean_text ：去除所有的控制符号，把连续的空格转化为单独的空格
+ handle_chinese_chars：给中文字的附近加上空格



复制 `bert-base-uncased` tokenizer

```python 
tokenizer.normalizer = normalizers.BertNormalizer(lowercase=True)
```

自己手动创建一个tokenizer

```python
tokenizer.normalizer = normalizers.Sequence(
    [normalizers.NFD(), normalizers.Lowercase(), normalizers.StripAccents()]
)
print(tokenizer.normalizer.normalize_str("Héllò hôw are ü?"))
hello how are u?
```



####  pre-tokenization step

可以直接使用BertPreTokenizer

```python
tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
```

也可以自己手动创建一个，Whitespace可以拆分空格和标点符号

```python
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
tokenizer.pre_tokenizer.pre_tokenize_str("Let's test my pre-tokenizer.")
[('Let', (0, 3)), ("'", (3, 4)), ('s', (4, 5)), ('test', (6, 10)), ('my', (11, 13)), ('pre', (14, 17)),
 ('-', (17, 18)), ('tokenizer', (18, 27)), ('.', (27, 28))]
```

如果只想直接分割空格

```python
pre_tokenizer = pre_tokenizers.Sequence(
    [pre_tokenizers.WhitespaceSplit(), pre_tokenizers.Punctuation()]
)
pre_tokenizer.pre_tokenize_str("Let's test my pre-tokenizer.")
[("Let's", (0, 5)), ('test', (6, 10)), ('my', (11, 13)), ('pre-tokenizer.', (14, 28))]
```



也可以将几个pre-tokenizers组合起来

```python
pre_tokenizer = pre_tokenizers.Sequence(
    [pre_tokenizers.WhitespaceSplit(), pre_tokenizers.Punctuation()]
)
pre_tokenizer.pre_tokenize_str("Let's test my pre-tokenizer.")

[('Let', (0, 3)), ("'", (3, 4)), ('s', (4, 5)), ('test', (6, 10)), ('my', (11, 13)), ('pre', (14, 17)),
 ('-', (17, 18)), ('tokenizer', (18, 27)), ('.', (27, 28))]
```



#### Running the input through the model 

上面的两步已经将模型初始化了，接下来就是训练模型，这里就需要WordPieceTrainer，需要把所有需要的特殊token传进去

```python
special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
trainer = trainers.WordPieceTrainer(vocab_size=25000, special_tokens=special_tokens)

# 使用之前定义的iterator来训练模型
tokenizer.model = models.WordPiece(unk_token="[UNK]")
tokenizer.train(["wikitext-2.txt"], trainer=trainer)
# 使用text文件来训练模型
tokenizer.model = models.WordPiece(unk_token="[UNK]")
tokenizer.train(["wikitext-2.txt"], trainer=trainer)
```

+  `vocab_size` 

+ `special_tokens`
+ `min_frequency`，token必须出现在词汇表中的次数
+  `continuing_subword_prefix` ,如果想要用##这个符号的话需要加上



使用encode()方法来测试tokenizer 

encoding实际上是拿到了所有输出需要的参数：`ids, type_ids, tokens, offsets, attention_mask, special_tokens_mask, and overflowing.`

```python 
encoding = tokenizer.encode("Let's test this tokenizer.")
print(encoding.tokens)
['let', "'", 's', 'test', 'this', 'tok', '##eni', '##zer', '.']

```



### post-processing

在开始处加上`[CLS]` token，结束的地方加上 `[SEP]` token，使用TemplateProcessor来处理这个问题，首先需要先区分对单个句子和对句子对的处理方式

对第一个句子， `$A`，第二个句子，$B

```python
cls_token_id = tokenizer.token_to_id("[CLS]")
sep_token_id = tokenizer.token_to_id("[SEP]")

print(cls_token_id, sep_token_id)
(2, 3)

#BERT定义的TemplateProcessor
tokenizer.post_processor = processors.TemplateProcessing(
    single=f"[CLS]:0 $A:0 [SEP]:0",
    pair=f"[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
    special_tokens=[("[CLS]", cls_token_id), ("[SEP]", sep_token_id)],
)
```



```python
encoding = tokenizer.encode("Let's test this tokenizer.")
print(encoding.tokens)
['[CLS]', 'let', "'", 's', 'test', 'this', 'tok', '##eni', '##zer', '.', '[SEP]']


encoding = tokenizer.encode("Let's test this tokenizer...", "on a pair of sentences.")
print(encoding.tokens)
print(encoding.type_ids)
['[CLS]', 'let', "'", 's', 'test', 'this', 'tok', '##eni', '##zer', '...', '[SEP]', 'on', 'a', 'pair', 'of', 'sentences', '.', '[SEP]']
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
```



#### 最后一步：decoder

```python 
tokenizer.decoder = decoders.WordPiece(prefix="##")
tokenizer.decode(encoding.ids)
"let's test this tokenizer... on a pair of sentences."


#把tokenizer以json的格式保存下来
tokenizer.save("tokenizer.json")

#reload tokenizer
new_tokenizer = Tokenizer.from_file("tokenizer.json")
```



#### 使用transformer里面的tokenizer

需要包裹在PreTrainedTokenizerFast里面

pass the tokenizer we built as a `tokenizer_object` or pass the tokenizer file we saved as `tokenizer_file`

 key thing：需要手动设置 special tokens

```python
from transformers import PreTrainedTokenizerFast

wrapped_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    # tokenizer_file="tokenizer.json", # You can load from the tokenizer file, alternatively
    unk_token="[UNK]",
    pad_token="[PAD]",
    cls_token="[CLS]",
    sep_token="[SEP]",
    mask_token="[MASK]",
)
```

