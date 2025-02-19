# 評估函式的實作

```python
from transformers import DataCollatorWithPadding
from datasets import load_dataset

#載入資料
dataset = load_dataset("csv", data_files="./ChnSentiCorp_htl_all.csv", split="train")
dataset

#==output==
Dataset({
    features: ['label', 'review'],
    num_rows: 7766
})


```

```python
#清理資料
cleared_dataset = dataset.filter(lambda item: item['review'] is not None )
cleared_dataset

#==output==
Dataset({
    features: ['label', 'review'],
    num_rows: 7765
})
```

```python
#拆分資料集
datasets = cleared_dataset.train_test_split(test_size=0.1)
datasets

#==output==
DatasetDict({
    train: Dataset({
        features: ['label', 'review'],
        num_rows: 6988
    })
    test: Dataset({
        features: ['label', 'review'],
        num_rows: 777
    })
})
```

```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-chinese")

def process_tokenizer(item):     
    '''
    只要先分詞,不要現在轉成tensor,也不用padding,轉成tensor和自動padding,由DataCollator來作
    '''
    
    tokenized:dict = tokenizer(item['review'],max_length=128, truncation=True)
    tokenized['label'] = item['label'] 
    return tokenized

tokenize_dataset = datasets.map(function=process_tokenizer,remove_columns=cleared_dataset.column_names,)
tokenize_dataset

#==output==
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-chinese")

def process_tokenizer(item):     
    '''
    只要先分詞,不要現在轉成tensor,也不用padding,轉成tensor和自動padding,由DataCollator來作
    '''
    
    tokenized:dict = tokenizer(item['review'],max_length=128, truncation=True)
    tokenized['label'] = item['label'] 
    return tokenized

tokenize_dataset = datasets.map(function=process_tokenizer,remove_columns=cleared_dataset.column_names,)
tokenize_dataset
```


```python
collator = DataCollatorWithPadding(tokenizer=tokenizer,return_tensors='pt')
collator
```

```python
from torch.utils.data import DataLoader
trainset, validset = tokenize_dataset['train'], tokenize_dataset['test']
trainloader = DataLoader(trainset, batch_size=32, collate_fn=collator, shuffle=True)
validloader = DataLoader(validset, batch_size=64, collate_fn=collator, shuffle=True)

next(enumerate(trainloader))[1]
```

```python
from torch.optim import Adam
from transformers import AutoModelForSequenceClassification
import torch

model = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-chinese", num_labels=2)
if torch.cuda.is_available():
    model = model.cuda()
optimizer = Adam(model.parameters(), lr=2e-5)
```

```python
import evaluate

#clf_metrics = evaluate.combine(['accuracy','f1'])
clf_metrics = evaluate.combine([
    'evaluate-main/metrics/accuracy/accuracy.py',
    'evaluate-main/metrics/f1/f1.py'])

```


```python
def evaluate()->float:
    model.eval() #讓模型進入評估模式
    for batch in validloader: #一次評估一個批次,目前每批次64筆
        if torch.cuda.is_available():
            batch = {k:v.cuda() for k,v in batch.items()}
        output = model(**batch)        
        pred = torch.argmax(output.logits,dim=-1) #pred是一維tensor() 
        clf_metrics.add_batch(predictions=pred.long(),references=batch['labels'].long()) 
    return clf_metrics.compute()
    
def train(epoch=3, log_step=100):
    global_step = 0
    for ep in range(epoch):
        model.train()
        for batch in trainloader:
            #batch是dict,包含input_ids,attention_mask,labels,token_type_ids
            if torch.cuda.is_available():
                batch = {k:v.cuda() for k,v in batch.items()}
            optimizer.zero_grad() #梯度歸零
            output = model(**batch) #前向傳播
            loss = output.loss #取得loss
            loss.backward()
            optimizer.step() #更新模型參數
            if global_step % log_step == 0: #每100個批次,輸出一次損失梯度
                print(f"第{ep+1}趟,執行第{global_step}批次,loss:{loss.item()}")
            global_step += 1
            
        #每訓練一趟就評估一次精準度
        clf = evaluate()
        print(f"第{ep+1}趟,{clf}")
        
train()

#==output==
第1趟,執行第0批次,loss:0.6826103925704956
第1趟,執行第100批次,loss:0.38519248366355896
第1趟,執行第200批次,loss:0.34996718168258667
第1趟,{'accuracy': 0.9099099099099099, 'f1': 0.9348230912476723}
第2趟,執行第300批次,loss:0.0774911567568779
第2趟,執行第400批次,loss:0.2131616622209549
第2趟,{'accuracy': 0.9060489060489061, 'f1': 0.9323447636700649}
第3趟,執行第500批次,loss:0.13012909889221191
第3趟,執行第600批次,loss:0.27839234471321106
第3趟,{'accuracy': 0.9047619047619048, 'f1': 0.9309701492537314}
```

