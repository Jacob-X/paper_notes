| Data format        | Loading script | Example                                                 |
| ------------------ | -------------- | ------------------------------------------------------- |
| CSV & TSV          | `csv`          | `load_dataset("csv", data_files="my_file.csv")`         |
| Text files         | `text`         | `load_dataset("text", data_files="my_file.txt")`        |
| JSON & JSON Lines  | `json`         | `load_dataset("json", data_files="my_file.jsonl")`      |
| Pickled DataFrames | `pandas`       | `load_dataset("pandas", data_files="my_dataframe.pkl")` |



对于不同的数据，有对应的dataload的方法

下面就是load json数据的方法

```python
from datasets import load_dataset

squad_it_dataset = load_dataset("json", data_files="SQuAD_it-train.json", field="data")

默认load进来的是dict格式的数据
squad_it_dataset
DatasetDict({
    train: Dataset({
        features: ['title', 'paragraphs'],
        num_rows: 442
    })
})
```



快速查看数据的内容

```python
squad_it_dataset["train"][0]

{
    "title": "Terremoto del Sichuan del 2008",
    "paragraphs": [
        {
            "context": "Il terremoto del Sichuan del 2008 o il terremoto...",
            "qas": [
                {
                    "answers": [{"answer_start": 29, "text": "2008"}],
                    "id": "56cdca7862d2951400fa6826",
                    "question": "In quale anno si è verificato il terremoto nel Sichuan?",
                },
                ...
            ],
        },
        ...
    ],
}
```





`data_files` argument that maps each split name

使用data_files参数把文件映射到对应的名称，这里是把train和test的文件都映射到对应的名称

```python
data_files = {"train": "SQuAD_it-train.json", "test": "SQuAD_it-test.json"}
squad_it_dataset = load_dataset("json", data_files=data_files, field="data")
squad_it_dataset

DatasetDict({
    train: Dataset({
        features: ['title', 'paragraphs'],
        num_rows: 442
    })
    test: Dataset({
        features: ['title', 'paragraphs'],
        num_rows: 48
    })
})
```





加载远程的数据

```python
例子，下载远程的数据并解压
!wget https://github.com/crux82/squad-it/raw/master/SQuAD_it-train.json.gz
!wget https://github.com/crux82/squad-it/raw/master/SQuAD_it-test.json.gz
!gzip -dkv SQuAD_it-*.json.gz

或者直接下载数据，这个方法返回的也是dict格式的数据
url = "https://github.com/crux82/squad-it/raw/master/"
data_files = {
    "train": url + "SQuAD_it-train.json.gz",
    "test": url + "SQuAD_it-test.json.gz",
}
squad_it_dataset = load_dataset("json", data_files=data_files, field="data")

```

