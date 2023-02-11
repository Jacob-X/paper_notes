在输入多条数据时，用作的前提条件是输入的tensor的ids必须是矩形的，但是由于输入内容的长短，在映射到ids的时候肯定会出现不一致的现象。

目前的解决方法就是做padding

把tensor填充到矩形

但是如果只对tensor进行填充，padding的ids在模型运算过程中是没有实际意义的，所以又增加了attention mask来提示模型的注意力机制

让模型不对padding的内容进行训练

```python 
batched_ids = [
    [200, 200, 200],
    [200, 200, tokenizer.pad_token_id],
]

attention_mask = [
    [1, 1, 1],
    [1, 1, 0],
]

outputs = model(torch.tensor(batched_ids), attention_mask=torch.tensor(attention_mask))

print(outputs.logits)

tensor([[ 1.5694, -1.3895],
        [ 0.5803, -0.4125]], grad_fn=<AddmmBackward>)
```



输入太长的解决方法

1. 使用支持长输入的模型
2. 截断输入的内容