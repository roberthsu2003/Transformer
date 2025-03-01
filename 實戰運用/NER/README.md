# 命名實體識別(Named Entity Recognition)
- token-classfication
- [實作ipynb](./ner1.ipynb)


### 1. 載入相關套件
- 必需安裝一個特別的評估套件seqeval
- [seqeval補充說明](./補充說明/seqeval說明.md)

```bash
pip intall seqeval
```

```python
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification
import evaluate
from datasets import load_dataset
```

### 2. 使用數據集

**下載資料**
- 由於目前ner的資料集非常少,先暫時使用中國的資料集

```python
#trust_remote_code,代表信任遠端資料,如不寫會有信任與否的提示
ner_datasets = load_dataset("peoples_daily_ner", cache_dir='./data', trust_remote_code=True)
ner_datasets

#==output==
DatasetDict({
    train: Dataset({
        features: ['id', 'tokens', 'ner_tags'],
        num_rows: 20865
    })
    validation: Dataset({
        features: ['id', 'tokens', 'ner_tags'],
        num_rows: 2319
    })
    test: Dataset({
        features: ['id', 'tokens', 'ner_tags'],
        num_rows: 4637
    })
})
```

**查看一筆資料

```python
ner_datasets['train'][0] # id,tokens,ner_tags

#==output==
{'id': '0',
 'tokens': ['海',
  '钓',
  '比',
  '赛',
  '地',
  '点',
  '在',
  '厦',
  '门',
  '与',
  '金',
  '门',
  '之',
  '间',
  '的',
  '海',
  '域',
  '。'],
 'ner_tags': [0, 0, 0, 0, 0, 0, 0, 5, 6, 0, 5, 6, 0, 0, 0, 0, 0, 0]}
```

**查看ner_tags,5,6所代表的意思**

- 透過classLabel,了解使用的標示格式是`IOB2`

```python
ner_datasets['train'].features

#==output==
{'id': Value(dtype='string', id=None),
 'tokens': Sequence(feature=Value(dtype='string', id=None), length=-1, id=None),
 'ner_tags': Sequence(feature=ClassLabel(names=['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC'], id=None), length=-1, id=None)}
```

**取出目前的標示的格式**
- `IOB2` 格式
- 人 -> `B-PER`
- 組織 -> `B-ORG`
- 地點 -> `B-LOC`

```python
#取得label_list
label_list = ner_datasets['train'].features['ner_tags'].feature.names
label_list

#==output==
['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC']
```

### 3. 數據預處理

**載入分詞器**

```python
tokenizer = AutoTokenizer.from_pretrained('google-bert/bert-base-chinese')
```

**不正確的寫法**

- 不正確,這是以字為單位,而不是以句子為單位

```python
#不正確,這是以字為單位,而不是以句子為單位
from pprint import pprint
pprint(tokenizer(ner_datasets['train'][0]['tokens']))

#==output==
{'attention_mask': [[1, 1, 1],
                    [1, 1, 1],
                    [1, 1, 1],
                    [1, 1, 1],
                    [1, 1, 1],
                    [1, 1, 1],
                    [1, 1, 1],
                    [1, 1, 1],
                    [1, 1, 1],
                    [1, 1, 1],
                    [1, 1, 1],
                    [1, 1, 1],
                    [1, 1, 1],
                    [1, 1, 1],
                    [1, 1, 1],
                    [1, 1, 1],
                    [1, 1, 1],
                    [1, 1, 1]],
 'input_ids': [[101, 3862, 102],
               [101, 7157, 102],
               [101, 3683, 102],
               [101, 6612, 102],
               [101, 1765, 102],
               [101, 4157, 102],
               [101, 1762, 102],
               [101, 1336, 102],
               [101, 7305, 102],
               [101, 680, 102],
               [101, 7032, 102],
               [101, 7305, 102],
               [101, 722, 102],
               [101, 7313, 102],
               [101, 4638, 102],
               [101, 3862, 102],
               [101, 1818, 102],
               [101, 511, 102]],
 'token_type_ids': [[0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0]]}
```

**解決方式,以句子為主的拆字**

```python
pprint(tokenizer(ner_datasets['train'][0]['tokens'],is_split_into_words=True))

#==output==
{'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
 'input_ids': [101,
               3862,
               7157,
               3683,
               6612,
               1765,
               4157,
               1762,
               1336,
               7305,
               680,
               7032,
               7305,
               722,
               7313,
               4638,
               3862,
               1818,
               511,
               102],
 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}
```

> 目前分詞後得到的資料有`input_ids`,`token_type_ids`,`attention_mask`
> 
> 所以目前還缺`labels`

**目前的問題**
- 無法確認中英文混合時的input_ids的數量
- 請看以下2個範例

```python
#全中文字
#文字字數8個加上101,102共10個
#input_ids = 101+字數+102,共10個
#所以labels也必需要10個
pprint(tokenizer(['小','明','愛','去','海','邊','釣','魚'],is_split_into_words=True))

#==output==
{'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
 'input_ids': [101, 2207, 3209, 2695, 1343, 3862, 6920, 7037, 7797, 102],
 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}
```

```python
#中英文混合
#文字字數8個加上101,102共10個
#input_ids = 101+10個字數+102,共12個 -> 英文字有sub_word的關係
#所以labels必需要12個
pprint(tokenizer(['小','明','interest','to','海','邊','釣','魚'],is_split_into_words=True))

#==output==
{'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
 'input_ids': [101,
               2207,
               3209,
               10673,
               12865,
               8415,
               8228,
               3862,
               6920,
               7037,
               7797,
               102],
 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}
```

> 問題就是目前的資料欄位`ner_tags`所提供的tags,無法直接給我要自行產生的`labels`使用,該如何解決

**問題**

```python
pprint(tokenizer(['小','明','interest','to','海','邊','釣','魚'],is_split_into_words=True))
ner_tags = [1, 2, 0, 0, 0, 0, 0, 0]
#input_ids = [101,2207,3209,10673,12865,8415,8228,3862,6920,7037,7797,102]
#101,102,要產生-100
#我們預計要產生的labels必需是=[-100,1,2,0,0,0,0,0,0,0,0,-100],共12,滿足input_ids的數量,但ner_tags欄位不可以滿足
#使用word_ids()解決問題
res = tokenizer(['小','明','interest','to','海','邊','釣','魚'],is_split_into_words=True)
print(res.word_ids())
#[None, 0, 1, 2, 2, 2, 3, 4, 5, 6, 7, None]
#None -> 101, 102
#0 -> '小'
#1 -> '明'
#2,2,2 -> 'interest'
#3 -> 'to'
#4 -> '海'
#5 -> '邊'
#6 -> '釣'
#7 -> '魚'
```

**解決方式**

```python
ner_tags = [1, 2, 0, 0, 0, 0, 0, 0] #自已手動的
res = tokenizer(['小','明','interest','to','海','邊','釣','魚'],max_length=128,truncation=True,is_split_into_words=True)
word_id = res.word_ids()
label_id = []
for id in word_id:
    if id is None:
        label_id.append(-100)
    else:
        #print(id)
        label_id.append(ner_tags[id])
print(label_id)

#==output==滿足我們的需求
[-100, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, -100]
```

**1.建立預處理的function->只1筆1筆處理資料**

```python
def process_one(example):
    ner_tags = example['ner_tags']
    res = tokenizer(example['tokens'],max_length=128, truncation=True, is_split_into_words=True)
    word_id = res.word_ids()
    label_id = []
    for id in word_id:
        if id is None:
            label_id.append(-100)
        else:
            label_id.append(ner_tags[id])
    res['labels'] = label_id

    return res
tokenized_datasets = ner_datasets.map(process_one)
print(tokenized_datasets)
pprint(tokenized_datasets['train'][0]['labels'])

#==output==
DatasetDict({
    train: Dataset({
        features: ['id', 'tokens', 'ner_tags', 'input_ids', 'token_type_ids', 'attention_mask', 'labels'],
        num_rows: 20865
    })
    validation: Dataset({
        features: ['id', 'tokens', 'ner_tags', 'input_ids', 'token_type_ids', 'attention_mask', 'labels'],
        num_rows: 2319
    })
    test: Dataset({
        features: ['id', 'tokens', 'ner_tags', 'input_ids', 'token_type_ids', 'attention_mask', 'labels'],
        num_rows: 4637
    })
})
[-100, 0, 0, 0, 0, 0, 0, 0, 5, 6, 0, 5, 6, 0, 0, 0, 0, 0, 0, -100]
```


**2.建立預處理的function->一次一個batch的資料**

```python
#一個批次處理的方式
from pprint import pprint
def process_batch(examples):
    tokenized_examples = tokenizer(examples['tokens'], max_length=128, truncation=True, is_split_into_words=True)
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_examples.word_ids(batch_index=i)
        label_ids = []
        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)
            else:
                label_ids.append(label[word_id])
        labels.append(label_ids)
    tokenized_examples['labels'] = labels

    return tokenized_examples

tokenized_datasets = ner_datasets.map(process_batch,batched=True)
print(tokenized_datasets)
print(tokenized_datasets['train'][0]['labels'])

#==output==
DatasetDict({
    train: Dataset({
        features: ['id', 'tokens', 'ner_tags', 'input_ids', 'token_type_ids', 'attention_mask', 'labels'],
        num_rows: 20865
    })
    validation: Dataset({
        features: ['id', 'tokens', 'ner_tags', 'input_ids', 'token_type_ids', 'attention_mask', 'labels'],
        num_rows: 2319
    })
    test: Dataset({
        features: ['id', 'tokens', 'ner_tags', 'input_ids', 'token_type_ids', 'attention_mask', 'labels'],
        num_rows: 4637
    })
})
[-100, 0, 0, 0, 0, 0, 0, 0, 5, 6, 0, 5, 6, 0, 0, 0, 0, 0, 0, -100]
```





