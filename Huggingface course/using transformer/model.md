# Huggingface Course



## model



Automodel这个类会自动猜测合适checkpoint的模型结构，然后使用这个结构来实例化模型



第一步，加载配置来实例化一个bert模型，但是这种没有经过训练的模型的效果是不好的

```python 
from transformers import BertConfig, BertModel

# Building the config
config = BertConfig()

# Building the model from the config
model = BertModel(config)

# bert模型配置文件里面的参数
print(config)

BertConfig {
  [...]
  "hidden_size": 768,
  "intermediate_size": 3072,
  "max_position_embeddings": 512,
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  [...]
}

```



可以使用 from_pretrained() 方法来加载经过预训练的模型

````python 
from transformers import BertModel

model = BertModel.from_pretrained("bert-base-cased")
````



使用 save_pretrained（） 方法来保存模型，保存了两个文件下来

+ json文件里面是模型架构的必要参数，也有一些原始的数据，比如transformer的版本，checkpoint的保存时间等
+ bin文件相当于是一个状态字典，里面保存了模型的weights，模型的超参数

```python
model.save_pretrained("directory_on_my_computer")

#保存了两个文件

config.json pytorch_model.bin

```



Transformer model只能处理数据，tokenizer把文字映射成数字（input IDs），

```python
sequences = ["Hello!", "Cool.", "Nice!"]

#tokenizer处理之后
encoded_sequences = [
    [101, 7592, 999, 102],
    [101, 4658, 1012, 102],
    [101, 3835, 999, 102],
]

# 转化为一个tensor作为模型的输出
import torch

model_inputs = torch.tensor(encoded_sequences)

# tensor可以作为模型的输入，模型可以接受许多的参数，但是只有tensor是必须输入的
output = model(model_inputs)

```

