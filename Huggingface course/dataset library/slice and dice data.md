# Slice Dataset



下载数据

```python
!wget "https://archive.ics.uci.edu/ml/machine-learning-databases/00462/drugsCom_raw.zip"
!unzip drugsCom_raw.zip


from datasets import load_dataset

data_files = {"train": "drugsComTrain_raw.tsv", "test": "drugsComTest_raw.tsv"}
# \t is the tab character in Python
以 \t 作为分隔符
drug_dataset = load_dataset("csv", data_files=data_files, delimiter="\t")

拿出一部分数据进行查看
seed=42,固定了随机种子，便于结果的复现
drug_sample = drug_dataset["train"].shuffle(seed=42).select(range(1000))
# Peek at the first few examples
drug_sample[:3]

```



读入的数据的结果

```python
{'Unnamed: 0': [87571, 178045, 80482],
 'drugName': ['Naproxen', 'Duloxetine', 'Mobic'],
 'condition': ['Gout, Acute', 'ibromyalgia', 'Inflammatory Conditions'],
 'review': ['"like the previous person mention, I&#039;m a strong believer of aleve, it works faster for my gout than the prescription meds I take. No more going to the doctor for refills.....Aleve works!"',
  '"I have taken Cymbalta for about a year and a half for fibromyalgia pain. It is great\r\nas a pain reducer and an anti-depressant, however, the side effects outweighed \r\nany benefit I got from it. I had trouble with restlessness, being tired constantly,\r\ndizziness, dry mouth, numbness and tingling in my feet, and horrible sweating. I am\r\nbeing weaned off of it now. Went from 60 mg to 30mg and now to 15 mg. I will be\r\noff completely in about a week. The fibro pain is coming back, but I would rather deal with it than the side effects."',
  '"I have been taking Mobic for over a year with no side effects other than an elevated blood pressure.  I had severe knee and ankle pain which completely went away after taking Mobic.  I attempted to stop the medication however pain returned after a few days."'],
 'rating': [9.0, 3.0, 10.0],
 'date': ['September 2, 2015', 'November 7, 2011', 'June 5, 2013'],
 'usefulCount': [36, 13, 128]}
```



存在几个问题

+ 第一列没有名字，像是病人的id

+ ```python
  测试一下unnamed一列是否是病人的id
  for split in drug_dataset.keys():
      assert len(drug_dataset[split]) == len(drug_dataset[split].unique("Unnamed: 0"))
  
  重命名unnamed
  drug_dataset = drug_dataset.rename_column(
      original_column_name="Unnamed: 0", new_column_name="patient_id"
  )
  drug_dataset
  
  DatasetDict({
      train: Dataset({
          features: ['patient_id', 'drugName', 'condition', 'review', 'rating', 'date', 'usefulCount'],
          num_rows: 161297
      })
      test: Dataset({
          features: ['patient_id', 'drugName', 'condition', 'review', 'rating', 'date', 'usefulCount'],
          num_rows: 53766
      })
  })
  ```

  

+ condition一栏的内容，存在大小写字母，字母形式不统一

```python 
map的时候不能有空值，所以在dataset.map()之前，需要先过滤掉condition为空的值

drug_dataset = drug_dataset.filter(lambda x: x["condition"] is not None)

def lowercase_condition(example):
    return {"condition": example["condition"].lower()}


drug_dataset.map(lowercase_condition)
# Check that lowercasing worked
drug_dataset["train"]["condition"][:3]
['left ventricular dysfunction', 'adhd', 'birth control']


```



+ review一栏中混杂了一些python的常见分隔符，\r，\n

​	首先创建了新的一栏review_length，

An alternative way to add new columns to a dataset is with the `Dataset.add_column()` function. This allows you to provide the column as a Python list or NumPy array and can be handy in situations where `Dataset.map()` is not well suited for your analysis.

```python 


def compute_review_length(example):
    return {"review_length": len(example["review"].split())}


drug_dataset = drug_dataset.map(compute_review_length)
# Inspect the first training example
drug_dataset["train"][0]

{'patient_id': 206461,
 'drugName': 'Valsartan',
 'condition': 'left ventricular dysfunction',
 'review': '"It has no side effect, I take it in combination of Bystolic 5 Mg and Fish Oil"',
 'rating': 9.0,
 'date': 'May 20, 2012',
 'usefulCount': 27,
 'review_length': 17}


sort方法进行排序
drug_dataset["train"].sort("review_length")[:3]

{'patient_id': [103488, 23627, 20558],
 'drugName': ['Loestrin 21 1 / 20', 'Chlorzoxazone', 'Nucynta'],
 'condition': ['birth control', 'muscle spasm', 'pain'],
 'review': ['"Excellent."', '"useless"', '"ok"'],
 'rating': [10.0, 1.0, 6.0],
 'date': ['November 4, 2008', 'March 24, 2017', 'August 20, 2016'],
 'usefulCount': [5, 2, 10],
 'review_length': [1, 1, 1]}


drug_dataset = drug_dataset.filter(lambda x: x["review_length"] > 30)
print(drug_dataset.num_rows)

{'train': 138514, 'test': 46108}

去除review中存在的一些html标识

import html
text = "I&#039;m a transformer called BERT"
html.unescape(text)
"I'm a transformer called BERT"

drug_dataset = drug_dataset.map(lambda x: {"review": html.unescape(x["review"])})

```



dataset.map()函数也可以使用多线程

```python
num_proc指定线程的数量

slow_tokenizer = AutoTokenizer.from_pretrained("bert-base-cased", use_fast=False)


def slow_tokenize_function(examples):
    return slow_tokenizer(examples["review"], truncation=True)


tokenized_dataset = drug_dataset.map(slow_tokenize_function, batched=True, num_proc=8)
```



将一个feature截断，然后得到两个特征，`return_overflowing_tokens=True`:可将截断的部分也返回

```python
def tokenize_and_split(examples):
    return tokenizer(
        examples["review"],
        truncation=True,
        max_length=128,
        return_overflowing_tokens=True,
    )

result = tokenize_and_split(drug_dataset["train"][0])
[len(inp) for inp in result["input_ids"]]
[128, 49]

截断feature之后，是会产生比原来更多的列的，不做处理就会报错，所以这里把原来的列名都删除了
tokenized_dataset = drug_dataset.map(
    tokenize_and_split, batched=True, remove_columns=drug_dataset["train"].column_names
)
len(tokenized_dataset["train"]), len(drug_dataset["train"])
(206772, 138514)



产生更多的列还有一种解决办法，就是把原来的列的数量大小变成处理之后的大小
def tokenize_and_split(examples):
    result = tokenizer(
        examples["review"],
        truncation=True,
        max_length=128,
        return_overflowing_tokens=True,
    )
    # Extract mapping between new and old indices
    sample_map = result.pop("overflow_to_sample_mapping")
    for key, values in examples.items():
        result[key] = [values[i] for i in sample_map]
    return result
tokenized_dataset = drug_dataset.map(tokenize_and_split, batched=True)
tokenized_dataset
DatasetDict({
    train: Dataset({
        features: ['attention_mask', 'condition', 'date', 'drugName', 'input_ids', 'patient_id', 'rating', 'review', 'review_length', 'token_type_ids', 'usefulCount'],
        num_rows: 206772
    })
    test: Dataset({
        features: ['attention_mask', 'condition', 'date', 'drugName', 'input_ids', 'patient_id', 'rating', 'review', 'review_length', 'token_type_ids', 'usefulCount'],
        num_rows: 68876
    })
})
```



transformer dataset有Dataset.set_format()方法，这个方法只会改变输出的dataset格式，

把dataset转化为pandas格式的

```python
drug_dataset.set_format("pandas")
这样取到的值就是pandas格式的了
drug_dataset["train"][:3]


可以用pandas来创建新列
train_df = drug_dataset["train"][:]
frequencies = (
    train_df["condition"]
    .value_counts()
    .to_frame()
    .reset_index()
    .rename(columns={"index": "condition", "condition": "frequency"})
)
frequencies.head()


```

![image-20230213185239019](http://picture.jacobx.top/markdown/image-20230213185239019.png)

```python
再从pandas的格式变化为dataset格式
from datasets import Dataset

freq_dataset = Dataset.from_pandas(frequencies)
freq_dataset

Dataset({
    features: ['condition', 'frequency'],
    num_rows: 819
})
```



创建一个validation set

```python
drug_dataset_clean = drug_dataset["train"].train_test_split(train_size=0.8, seed=42)
# Rename the default "test" split to "validation"
drug_dataset_clean["validation"] = drug_dataset_clean.pop("test")
# Add the "test" set to our `DatasetDict`
drug_dataset_clean["test"] = drug_dataset["test"]
drug_dataset_clean


DatasetDict({
    train: Dataset({
        features: ['patient_id', 'drugName', 'condition', 'review', 'rating', 'date', 'usefulCount', 'review_length', 'review_clean'],
        num_rows: 110811
    })
    validation: Dataset({
        features: ['patient_id', 'drugName', 'condition', 'review', 'rating', 'date', 'usefulCount', 'review_length', 'review_clean'],
        num_rows: 27703
    })
    test: Dataset({
        features: ['patient_id', 'drugName', 'condition', 'review', 'rating', 'date', 'usefulCount', 'review_length', 'review_clean'],
        num_rows: 46108
    })
})
```

