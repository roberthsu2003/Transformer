# Model的預訓練
- 僅使用tokenizer和model
- 使用原生pytorch做預訓練
- 複雜度較高
- 當使用huggingface的dataset,evaluate,trainer,可以簡單化目前的流程

## 使用的資料集
- [SophonPlus](https://github.com/SophonPlus/ChineseNlpCorpus)
- 資料夾內有提供轉換為繁體中文的檔案(ChnSentiCorp_htl_all.csv)

## 載入數據

```python
#載入數據
import pandas as pd

data = pd.read_csv('./ChnSentiCorp_htl_all.csv')
data.head()
```

**清理數據**

```python
#清理數據
data = data.dropna()
data.info()

#==output==
<class 'pandas.core.frame.DataFrame'>
Index: 7765 entries, 0 to 7765
Data columns (total 2 columns):
 #   Column  Non-Null Count  Dtype 
---  ------  --------------  ----- 
 0   label   7765 non-null   int64 
 1   review  7765 non-null   object
dtypes: int64(1), object(1)
memory usage: 182.0+ KB
```

**取出第1筆資料**

```python
data.iloc[0]['review'], data.iloc[0]['label']

#==output==
('距離川沙公路較近,但是公交指示不對,如果是"蔡陸線"的話,會非常麻煩.建議用別的路線.房間較為簡單.', np.int64(1))
```

## 建立pytorch的Dataset
[**pytorch DataSet的簡單範例**](./pytorch_dataset.md)

**建立自訂Dataset類別**

```
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.data = pd.read_csv('./ChnSentiCorp_htl_all.csv')
        self.data = self.data.dropna()

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data.iloc[index]['review'], self.data.iloc[index]['label']
```

**取出前5筆資料**

```python
dataset = MyDataset()
for index in range(5):
    print(dataset[i])
```

**切割資料集,分為訓練用和驗証用**

```python
from torch.utils.data import random_split
dataset = MyDataset()
trainset, validset = random_split(dataset, lengths=[0.9, 0.1]) #lengths是比例,2個加總必需是1,會隨機分配,不會依照順序
print(trainset)

len(trainset), len(validset)

#==output==
<torch.utils.data.dataset.Subset object at 0xffff1c4e54c0>
(6989, 776)
```

## 建立DataLoader
- dataloader可以使用批次方式載入資料
- 有數值的部份會自動轉成tensor
- 文字部份不會自動轉成tensor
- [DataLoader範例](./dataloader.md)

```python
from torch.utils.data import DataLoader
import torch

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-chinese")

def collate_func(batch): #batch整個32筆資料,重整32筆資料的內容,傳出tuple
    texts, labels = [],[]
    for item in batch:
        texts.append(item[0])
        labels.append(item[1])
    inputs = tokenizer(texts, max_length=128, padding="max_length", truncation=True,return_tensors='pt')
    #print(inputs)
    inputs['labels'] = torch.tensor(labels)
    return inputs

trainloader = DataLoader(trainset, batch_size=32, shuffle=True,collate_fn=collate_func) #collate_fn,建立處理batch內的資料
validloader = DataLoader(validset, batch_size=64, shuffle=False,collate_fn=collate_func)

next(enumerate(trainloader))[1]

```

## 建立模型和優化器

- [Adam優化器說明](./adam.md)

```python
from torch.optim import Adam
model = AutoModelForSequenceClassification.from_pretrained('google-bert/bert-base-chinese')
if torch.cuda.is_available():
    model = model.cuda()
optimizer = Adam(model.parameters(), lr=2e-5)
```

## 訓練和驗証

```
與反向傳播 (backward()) 的關係

zero_grad() 和 backward() 方法在模型的訓練過程中是緊密配合使用的：

- zero_grad() 負責清空之前的梯度。 它確保在計算新梯度之前，梯度緩衝區是乾淨的。
- backward() 負責計算當前批次的梯度，並將這些梯度累積到模型參數的 .grad 屬性中。 它計算出模型參數應該如何調整才能減少損失。
- step() 負責使用當前累積的梯度來更新模型參數。 優化器 (例如 Adam) 根據 backward() 計算出的梯度，以及優化器自身的算法 (例如 Adam 的自適應學習率機制)，來決定如何調整模型參數。
```

```
def evaluate():
    model.eval()
    acc_num = 0
    with torch.inference_mode():
       for batch in validloader:
            if torch.cuda.is_available():
                batch = {k: v.cuda() for k, v in batch.items()} 
            output = model(**batch)
            pred = torch.argmax(output.logits, dim=-1)
            acc_num += (pred.long() == batch["labels"].long()).float().sum()
    return acc_num / len(validset)


def train(epoch=3, log_step=100):
    global_step = 0
    for ep in range(epoch):
        model.train()
        for batch in trainloader:
            if torch.cuda.is_available():
                batch = {k: v.cuda() for k, v in batch.items()}        
            optimizer.zero_grad() #模型參數的梯度歸零
            output = model(**batch)
            output.loss.backward() #計算損失梯度
            optimizer.step() #更新模型參數
            if global_step % log_step == 0:
                print(f'ep:{ep}, global_step:{global_step},loss:{output.loss.item()}')
            global_step += 1
        acc = evaluate()
        print(f"ep:{ep}, acc:{acc}")
train()
```

## 儲存model和tokenizer至本機資料夾

```python
model.save_pretrained('./saved_model')
tokenizer.save_pretrained('./saved_model')
```

## 由本機端載入,並預測

```
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
id2_label = {0:"差評!", 1:"好評!"}
model = AutoModelForSequenceClassification.from_pretrained('./saved_model')
tokenizer = AutoTokenizer.from_pretrained('./saved_model')
sen="服務人員臉色不好看"
model.eval()
with torch.inference_mode():
    inputs = tokenizer(sen,return_tensors='pt')
    logits = model(**inputs).logits
    pred = torch.argmax(logits,dim=-1)
    print(f"輸入:{sen}\n模型預測結果:{id2_label.get(pred.item())}")

#==output==
輸入:服務人員臉色不好看
模型預測結果:差評!
```

## 上傳model和tokenizer至huggingface

```python
from huggingface_hub import login, HfApi
login()
```

```python
repo_id = "roberthsu2003/save_model"
model.push_to_hub(repo_id)
tokenizer.push_to_hub(repo_id)
```

## 由huggingface下載至本地端預測

```python
# Use a pipeline as a high-level helper
from transformers import pipeline
id2str = {'LABEL_0':"差評",'LABEL_1':"好評"}
pipe = pipeline("text-classification", model="roberthsu2003/save_model")
sen="服務人員很熱心"
output = pipe(sen)
key = output[0]['label']
score_str = id2str.get(key)
score = output[0]['score']
print(f"{score_str},{score}")

#==output==
Device set to use cpu
好評,0.9752063751220703
```




