# fine-tuning model



### data pro-processing

 首先是数据预处理的部分，这是一章的内容

```python
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

raw_datasets = load_dataset("glue", "mrpc")
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)


tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
```



### training

定义trainer之前，需要先定义 TrainingArguments类 ，这个类里面包含了trainer在训练过程中需要使用到的超参数（hyperparameters ）

TrainingArguments（）这个函数，只需要提供保存当前模型的存储位置

```python
from transformers import TrainingArguments

training_args = TrainingArguments("test-trainer")
```



第二步是定义模型，这里用的是一个序列分类的任务

```python
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
```



第三步，构建自己的trainer

```python 
from transformers import Trainer

trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)

在dataset 上 fine tune model
trainer.train()
```



### Evaluation

构建一个compute_metrics()

+ 接收一个EvalPrediction object（一个由预测字段和label_ids字段组成的元组）
+ 返回一个有字符和float组成的字典，预测的对象和预测的值



```python
predictions = trainer.predict(tokenized_datasets["validation"])
print(predictions.predictions.shape, predictions.label_ids.shape)

predictions是两维的数组，408 是数据集中我们使用的元素个数
(408, 2) (408,)
```



trainer.predict（）输出的是：predictions, label_ids, and metrics

metrics 字段包含了模型在dataset上计算得到的loss

predictions 408*2 ，408是数据集中的元素个数，2是模型返回的类别的logits，为了将logits变换成我们可以比较的标签，我们需要在第二个轴上取最大值的索引，就是2个logits中取最大的那个

```python
import numpy as np

preds = np.argmax(predictions.predictions, axis=-1)
```



构建evaluate函数

```python
import evaluate

metric = evaluate.load("glue", "mrpc")
metric.compute(predictions=preds, references=predictions.label_ids)
{'accuracy': 0.8578431372549019, 'f1': 0.8996539792387542}
```





一个完整的compute_metrics（）函数

```python
def compute_metrics(eval_preds):
    metric = evaluate.load("glue", "mrpc")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)
```



定义一个有compute_metrics的trainer，评估策略：每一个epoch结束之后进行loss和metric的计算

```python
training_args = TrainingArguments("test-trainer", evaluation_strategy="epoch")
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)
```

