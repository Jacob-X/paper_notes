tokenizer中包含了之前说的所有的功能，padding，attention mask

````python 
sequences = ["I've been waiting for a HuggingFace course my whole life.", "So have I!"]

model_inputs = tokenizer(sequences)

# Will pad the sequences up to the maximum sequence length
model_inputs = tokenizer(sequences, padding="longest")

# Will pad the sequences up to the model max length
# (512 for BERT or DistilBERT)
model_inputs = tokenizer(sequences, padding="max_length")

# Will pad the sequences up to the specified max length
model_inputs = tokenizer(sequences, padding="max_length", max_length=8)


return_tensors="pt”,Returns PyTorch tensors
model_inputs = tokenizer(sequences, padding=True, return_tensors="pt")


````



在使用tokenizer的时候，实际上是增加101和102的id，用来标识句子的开始和结尾，便于句子的训练

```python 
model_inputs = tokenizer(sequence)
print(model_inputs["input_ids"])

[101, 1045, 1005, 2310, 2042, 3403, 2005, 1037, 17662, 12172, 2607, 2026, 2878, 2166, 1012, 102]

print(tokenizer.decode(model_inputs["input_ids"]))
"[CLS] i've been waiting for a huggingface course my whole life. [SEP]"
```



一个完整的输入过程

```python 
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
sequences = ["I've been waiting for a HuggingFace course my whole life.", "So have I!"]

tokens = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")
output = model(**tokens)
```

