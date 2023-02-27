# Training a new tokenizer from an old one

为了适应自己的数据语料，新训练一个tokenizer

训练tokenzier和训练model是不一样的

+ 训练模型使用的是随机梯度下降，难以得到完全相同的结果

+ 训练tokenizer是一个统计的过程，从语料中挑选出效果最好的子词，使用算法的标准的规则去挑选子词，相同的算法和语料，得到的子词应该是相同的。

train a new tokenizer with the same characteristics as an existing one: `AutoTokenizer.train_new_from_iterator()`,这个方法从已经存在的tokenizer中训练出一个具有相同特征的新的tokenizer



例子，训练一个针对python代码的tokenizer

```python 
from datasets import load_dataset

#code_search_net是一个python代码的数据库

# This can take a few minutes to load, so grab a coffee or tea while you wait!
raw_datasets = load_dataset("code_search_net", "python")

raw_datasets["train"]

Dataset({
    features: ['repository_name', 'func_path_in_repository', 'func_name', 'whole_func_string', 'language', 
      'func_code_string', 'func_code_tokens', 'func_documentation_string', 'func_documentation_tokens', 'split_name', 
      'func_code_url'
    ],
    num_rows: 412178
})


#从这个数据集里面抽取完整的函数代码来进行训练
print(raw_datasets["train"][123456]["whole_func_string"])
def handle_simple_responses(
      self, timeout_ms=None, info_cb=DEFAULT_MESSAGE_CALLBACK):
    """Accepts normal responses from the device.

    Args:
      timeout_ms: Timeout in milliseconds to wait for each response.
      info_cb: Optional callback for text sent from the bootloader.

    Returns:
      OKAY packet's message.
    """
    return self._accept_responses('OKAY', info_cb, timeout_ms=timeout_ms)
```



训练之前的第一步是把数据放到一个list of texts的迭代器里，使用list可以加速tokenizer，避免把所有的数据一次性加载到内存中



在数据集很小的情况下，可以使用这样的方式

```python
# Don't uncomment the following line unless your dataset is small!
training_corpus = [raw_datasets["train"][i: i + 1000]["whole_func_string"] for i in range(0, len(raw_datasets["train"]), 1000)]
```



使用generator可以避免一次性把所有的数据都读到内存中

python generator是一边循环一边计算的机制

```python
下面这个generator就是一次只加载1000条数据进来计算，需要的时候再继续加载1000条数据进来
training_corpus = (
    raw_datasets["train"][i : i + 1000]["whole_func_string"]
    for i in range(0, len(raw_datasets["train"]), 1000)
)
```



generator的一个例子

```python
gen = (i for i in range(10))
print(list(gen))
print(list(gen))
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
[]
```



最后使用的函数

```python
def get_training_corpus():
    return (
        raw_datasets["train"][i : i + 1000]["whole_func_string"]
        for i in range(0, len(raw_datasets["train"]), 1000)
    )

training_corpus = get_training_corpus()
```



可以使用更复杂逻辑的形式

```python
def get_training_corpus():
    dataset = raw_datasets["train"]
    for start_idx in range(0, len(dataset), 1000):
        samples = dataset[start_idx : start_idx + 1000]
        yield samples["whole_func_string"]
```



### 训练一个新的tokenizer

先把旧的tokenizer加载进来

```python 
from transformers import AutoTokenizer

old_tokenizer = AutoTokenizer.from_pretrained("gpt2")
```

新训练的tokenizer在算法方面和之前的tokenizer是一样的，只是在词汇方面有不同

旧的tokenizer

```python 
example = '''def add_numbers(a, b):
    """Add the two numbers `a` and `b`."""
    return a + b'''

tokens = old_tokenizer.tokenize(example)

tokens
['def', 'Ġadd', '_', 'n', 'umbers', '(', 'a', ',', 'Ġb', '):', 'Ċ', 'Ġ', 'Ġ', 'Ġ', 'Ġ"""', 'Add', 'Ġthe', 'Ġtwo',
 'Ġnumbers', 'Ġ`', 'a', '`', 'Ġand', 'Ġ`', 'b', '`', '."', '""', 'Ċ', 'Ġ', 'Ġ', 'Ġ', 'Ġreturn', 'Ġa', 'Ġ+', 'Ġb']
```



使用`train_new_from_iterator()`来新训练一个tokenizer

```python 
tokenizer = old_tokenizer.train_new_from_iterator(training_corpus, 52000)

tokens = tokenizer.tokenize(example)
tokens

['def', 'Ġadd', '_', 'numbers', '(', 'a', ',', 'Ġb', '):', 'ĊĠĠĠ', 'Ġ"""', 'Add', 'Ġthe', 'Ġtwo', 'Ġnumbers', 'Ġ`',
 'a', '`', 'Ġand', 'Ġ`', 'b', '`."""', 'ĊĠĠĠ', 'Ġreturn', 'Ġa', 'Ġ+', 'Ġb']
```

这个方法有一个先决条件，就是只有使用fast tokenizer的时候`train_new_from_iterator()`才会work

tokenizer有两种：单纯使用python写的tokenzier和使用别的较为高效的语言(Rust)编写的tokenizer，单纯使用python编写的tokenizer的速度是非常慢的，

Tokenizers library里面集成了这些高效的tokenizer,`AutoTokenizer` API会优先选择速度最快的tokenizer



### Saving the tokenizer

```python 
tokenizer.save_pretrained("code-search-net-tokenizer")
```

