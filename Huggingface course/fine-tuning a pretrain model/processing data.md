# processing data



从hugging face上加载数据

```python 
from datasets import load_dataset

raw_datasets = load_dataset("glue", "mrpc")
raw_datasets
```



在加载句子对的数据的时候，有顺序的关系

```python 
inputs = tokenizer("This is the first sentence.", "This is the second one.")
inputs
{ 
  'input_ids': [101, 2023, 2003, 1996, 2034, 6251, 1012, 102, 2023, 2003, 1996, 2117, 2028, 1012, 102],
  'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
  'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
}

tokenizer.convert_ids_to_tokens(inputs["input_ids"])
['[CLS]', 'this', 'is', 'the', 'first', 'sentence', '.', '[SEP]', 'this', 'is', 'the', 'second', 'one', '.', '[SEP]']
[CLS] sentence1 [SEP] sentence2 [SEP]
['[CLS]', 'this', 'is', 'the', 'first', 'sentence', '.', '[SEP]', 'this', 'is', 'the', 'second', 'one', '.', '[SEP]']
[      0,      0,    0,     0,       0,          0,   0,       0,      1,    1,     1,        1,     1,   1,       1]
```

token_type_ids，就是用来标识第一句和第二句的位置关系的，bert里面有，别的模型里面不一定有，应为别的模型里面不一定有next sentence prediction 的任务



向tokenizer中加载一系列的输入数据

数据的格式如下

```python 
DatasetDict({
    train: Dataset({
        features: ['sentence1', 'sentence2', 'label', 'idx'],
        num_rows: 3668
    })
    validation: Dataset({
        features: ['sentence1', 'sentence2', 'label', 'idx'],
        num_rows: 408
    })
    test: Dataset({
        features: ['sentence1', 'sentence2', 'label', 'idx'],
        num_rows: 1725
    })
})

raw_train_dataset.features
{'sentence1': Value(dtype='string', id=None),
 'sentence2': Value(dtype='string', id=None),
 'label': ClassLabel(num_classes=2, names=['not_equivalent', 'equivalent'], names_file=None, id=None),
 'idx': Value(dtype='int32', id=None)}
```



这个方法对运行内存的要求特别高

```python 
tokenized_dataset = tokenizer(
    raw_datasets["train"]["sentence1"],
    raw_datasets["train"]["sentence2"],
    padding=True,
    truncation=True,
)
```

 Dataset.map() 的方法

```python
def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

example: Dict[str, Any]
输入一个dict（example）,然后返回一个新的dict,包括了input_ids, attention_mask, 和token_type_ids
当dict中包含许多的样本时（list of sentence），batched = True就可以了，这样就可以批量的处理list数据了，大大加快了tokenization的时间
只有把输入数据一次性都载入的时候，这样的处理方式才是最快的

#调用的方式，tokenize_function是上面编写的处理函数
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
tokenized_datasets
```



对数据预处理之后，就多了许多模型需要的features

预处理数据的时候可以使用num_proc来多线程

```python
DatasetDict({
    train: Dataset({
        features: ['attention_mask', 'idx', 'input_ids', 'label', 'sentence1', 'sentence2', 'token_type_ids'],
        num_rows: 3668
    })
    validation: Dataset({
        features: ['attention_mask', 'idx', 'input_ids', 'label', 'sentence1', 'sentence2', 'token_type_ids'],
        num_rows: 408
    })
    test: Dataset({
        features: ['attention_mask', 'idx', 'input_ids', 'label', 'sentence1', 'sentence2', 'token_type_ids'],
        num_rows: 1725
    })
})
```



### Dynamic padding

在预处理数据的时候最好不要在过长的句子后面加padding

只在需要的batch上加padding，有效的加快训练速度



把所有sample放到一个batch的函数叫做 collate function，是构建DataLoader的时候的一个参数，可以把sample链接起来

DataCollatorWithPadding

动态padding，就是count每一个输入的长度，然后将所有的输入都padding到当前最长的那条输入的长度



```python 
from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

samples = tokenized_datasets["train"][:8]
samples = {k: v for k, v in samples.items() if k not in ["idx", "sentence1", "sentence2"]}
[len(x) for x in samples["input_ids"]]
[50, 59, 47, 67, 59, 50, 62, 32]

batch = data_collator(samples)
{k: v.shape for k, v in batch.items()}

batch里面的格式
{'attention_mask': torch.Size([8, 67]),
 'input_ids': torch.Size([8, 67]),
 'token_type_ids': torch.Size([8, 67]),
 'labels': torch.Size([8])}

```

