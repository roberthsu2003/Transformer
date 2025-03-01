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

### 4. 建立模型

```python
#label_list
#['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC']
model = AutoModelForTokenClassification.from_pretrained("google-bert/bert-base-chinese",num_labels=len(label_list)) #預設是2個,現在是7個
```

### 5. 建立評估function

```python
seqeval = evaluate.load('seqeval')
seqeval #查看所需要的說明書

#==output==
EvaluationModule(name: "seqeval", module_type: "metric", features: {'predictions': Sequence(feature=Value(dtype='string', id='label'), length=-1, id='sequence'), 'references': Sequence(feature=Value(dtype='string', id='label'), length=-1, id='sequence')}, usage: """
Produces labelling scores along with its sufficient statistics
from a source against one or more references.

Args:
    predictions: List of List of predicted labels (Estimated targets as returned by a tagger)
    references: List of List of reference labels (Ground truth (correct) target values)
    suffix: True if the IOB prefix is after type, False otherwise. default: False
    scheme: Specify target tagging scheme. Should be one of ["IOB1", "IOB2", "IOE1", "IOE2", "IOBES", "BILOU"].
        default: None
    mode: Whether to count correct entity labels with incorrect I/B tags as true positives or not.
        If you want to only count exact matches, pass mode="strict". default: None.
    sample_weight: Array-like of shape (n_samples,), weights for individual samples. default: None
    zero_division: Which value to substitute as a metric value when encountering zero division. Should be on of 0, 1,
        "warn". "warn" acts as 0, but the warning is raised.

Returns:
    'scores': dict. Summary of the scores for overall and per type
        Overall:
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': F1 score, also known as balanced F-score or F-measure,
        Per type:
            'precision': precision,
            'recall': recall,
            'f1': F1 score, also known as balanced F-score or F-measure
Examples:

    >>> predictions = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
    >>> references = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
    >>> seqeval = evaluate.load("seqeval")
    >>> results = seqeval.compute(predictions=predictions, references=references)
    >>> print(list(results.keys()))
    ['MISC', 'PER', 'overall_precision', 'overall_recall', 'overall_f1', 'overall_accuracy']
    >>> print(results["overall_f1"])
    0.5
    >>> print(results["PER"]["f1"])
    1.0
""", stored examples: 0)
```

```python
import numpy as np

def eval_metric(pred):
    predictions, labels = pred
    #print(predictions),評估時print()可以比較了解
    predictions = np.argmax(predictions, axis=-1) #變為和label一樣的一維資料

    #刪除-100
    truth_predictions = [
        [label_list[p] for p,l in zip(prediction, label) if p != -100]
        for prediction, label in zip(predictions, labels)
    ]

    truth_labels = [
        [label_list[l] for p,l in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    result = seqeval.compute(predictions=truth_predictions, references=truth_labels, mode='strict', scheme='IOB2')
    return{
        "f1": result['overall_f1']
    }
```


### 6. 配置訓練參數

```python
args = TrainingArguments(
    output_dir = "models_for_ner",
    per_device_eval_batch_size=64,
    per_device_train_batch_size=128,
    eval_strategy='epoch',
    save_strategy='epoch',
    metric_for_best_model='f1',
    load_best_model_at_end=True,
    logging_steps = 50,
    num_train_epochs=3,
    report_to='none'
)
```

### 7. 建立訓練器

```python
trainer = Trainer(
    model = model,
    args = args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation'],
    compute_metrics=eval_metric,
    data_collator=DataCollatorForTokenClassification(tokenizer=tokenizer)
)
```

```python
trainer.train()
```

### 8. 評估模型

```python
trainer.evaluate(eval_dataset=tokenized_datasets['test'])

#==output==
{'eval_loss': 0.02187521383166313,
 'eval_f1': 0.9508438253415484,
 'eval_runtime': 36.0562,
 'eval_samples_per_second': 128.605,
 'eval_steps_per_second': 1.026,
 'epoch': 3.0}
```

### 9. 上傳模型和所有評估資料

```python
from huggingface_hub import login
login()
```

```python
trainer.push_to_hub("roberthsu2003") #由於有設./checkpoints,所以自動產生checkpoints的repo,也會自動上傳評估至repo
#同時要上傳tokenizer
model_name = "roberthsu2003/models_for_ner"
tokenizer.push_to_hub(model_name)
```

### 10. 使用訓練中的模型

- `LABEL_1`,不是我們要使用的

```python
ner_pipe = pipeline("token-classification", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
ner_pipe("徐國堂在台北上班")

#==output==
[{'entity_group': 'LABEL_1',
  'score': np.float32(0.99952245),
  'word': '徐',
  'start': 0,
  'end': 1},
 {'entity_group': 'LABEL_2',
  'score': np.float32(0.9994726),
  'word': '國 堂',
  'start': 1,
  'end': 3},
 {'entity_group': 'LABEL_0',
  'score': np.float32(0.99968326),
  'word': '在',
  'start': 3,
  'end': 4},
 {'entity_group': 'LABEL_5',
  'score': np.float32(0.99884737),
  'word': '台',
  'start': 4,
  'end': 5},
 {'entity_group': 'LABEL_6',
  'score': np.float32(0.99812204),
  'word': '北',
  'start': 5,
  'end': 6},
 {'entity_group': 'LABEL_0',
  'score': np.float32(0.99980557),
  'word': '上 班',
  'start': 6,
  'end': 8}]
```

```python
print(model.config.id2label)

#==output==
{0: 'LABEL_0', 1: 'LABEL_1', 2: 'LABEL_2', 3: 'LABEL_3', 4: 'LABEL_4', 5: 'LABEL_5', 6: 'LABEL_6'}
```

**修改model.config.id2label**

```python
label_list = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC']
model.config.id2label = {idx: label for idx, label in enumerate(label_list)}
model.config.id2label

#==output==
{0: 'O',
 1: 'B-PER',
 2: 'I-PER',
 3: 'B-ORG',
 4: 'I-ORG',
 5: 'B-LOC',
 6: 'I-LOC'}
```

### 重新上傳(只上傳模型,沒有上傳評估資料和README.md)

```python
from huggingface_hub import login
login()
```

```
model.push_to_hub(model_name)
```

### 使用HuggingFace上的模型(pipeline方式)

- pipeline()->使用NER的官方說明

https://huggingface.co/docs/transformers/main_classes/pipelines#transformers.TokenClassificationPipeline


```python
ner_pipe = pipeline("token-classification", model='roberthsu2003/models_for_ner',aggregation_strategy="simple")
inputs = "徐國堂在台北上班" 
res = ner_pipe(inputs)

#解決文字中間有空格的問題
ner_result = {}
for r in res:
  if r["entity_group"] not in ner_datasets:
    ner_result[r['entity_group']] = []
  ner_result[r['entity_group']].append(inputs[r['start']:r['end']])
ner_result

#==output==
{'PER': ['徐國堂'], 'LOC': ['台北']}
```


### 使用HuggingFace上的模型(model,tokenizer方式)

```python
#使用model,tokenizer的使用方法
from transformers import AutoModelForTokenClassification, AutoTokenizer
import numpy as np

# Load the pre-trained model and tokenizer
model = AutoModelForTokenClassification.from_pretrained('roberthsu2003/models_for_ner')
tokenizer = AutoTokenizer.from_pretrained('roberthsu2003/models_for_ner')

# The label mapping (you might need to adjust this based on your training)
label_list = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC']

def predict_ner(text):
    """Predicts NER tags for a given text using the loaded model."""
    # Encode the text
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    
    # Get model predictions
    outputs = model(**inputs)
    predictions = np.argmax(outputs.logits.detach().numpy(), axis=-1)
    
    # Get the word IDs from the encoded inputs
    # This is the key change - word_ids() is a method on the encoding result, not the tokenizer itself
    word_ids = inputs.word_ids(batch_index=0)
    
    pred_tags = []
    for word_id, pred in zip(word_ids, predictions[0]):
        if word_id is None:
            continue  # Skip special tokens
        pred_tags.append(label_list[pred])

    return pred_tags

#To get the entities, you'll need to group consecutive non-O tags:

def get_entities(tags):
    """Groups consecutive NER tags to extract entities."""
    entities = []
    start_index = -1
    current_entity_type = None
    for i, tag in enumerate(tags):
        if tag != 'O':
            if start_index == -1:
                start_index = i
                current_entity_type = tag[2:] # Extract entity type (e.g., PER, LOC, ORG)
        else: #tag == 'O'
            if start_index != -1:
                entities.append((start_index, i, current_entity_type))
                start_index = -1
                current_entity_type = None
    if start_index != -1:
        entities.append((start_index, len(tags), current_entity_type))
    return entities

# Example usage:
text = "徐國堂在台北上班"
ner_tags = predict_ner(text)
print(f"Text: {text}")
#==output==
#Text: 徐國堂在台北上班
print(f"NER Tags: {ner_tags}")
#===output==
#NER Tags: ['B-PER', 'I-PER', 'I-PER', 'O', 'B-LOC', 'I-LOC', 'O', 'O']


entities = get_entities(ner_tags)
word_tokens = tokenizer.tokenize(text)  # Tokenize to get individual words
print(f"Entities:")
for start, end, entity_type in entities:
    entity_text = "".join(word_tokens[start:end])
    print(f"- {entity_text}: {entity_type}")

#==output==
#Entities:
#- 徐國堂: PER
#- 台北: LOC

```

















