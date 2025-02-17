# 實作-使用Huggingface Dataset,取代pytorch Dataset

```
#載入數據
from transformers import DataCollatorWithPadding
from datasets import load_dataset

#載入資料
dataset = load_dataset("csv",data_files="./ChnSentiCorp_htl_all.csv",split="train")
dataset

#==output==
Dataset({
    features: ['label', 'review'],
    num_rows: 7766
})
```

```python
#清理資料
cleared_dataset = dataset.filter(lambda item: item['review'] is not None)
cleared_dataset

#==output==
Dataset({
    features: ['label', 'review'],
    num_rows: 7765
})
```

```python
#拆分資料集
from datasets import Dataset
datasets = cleared_dataset.train_test_split(train_size=0.9,test_size=0.1)
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

**將DataSet分詞**

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('google-bert/bert-base-chinese')
def process_tokenizer(example:dict)->dict:
    '''
    只要先分詞,不要現在轉成tensor,轉成tensor,由DataCollator來作
    '''
    tokenized = tokenizer(example['review'],max_length=128,truncation=True)
    tokenized['label'] = example['label']
    return tokenized


tokenizer_dataset = datasets.map(function=process_tokenizer,remove_columns=cleared_dataset.column_names)
tokenizer_dataset

#==output==
DatasetDict({
    train: Dataset({
        features: ['label', 'input_ids', 'token_type_ids', 'attention_mask'],
        num_rows: 6988
    })
    test: Dataset({
        features: ['label', 'input_ids', 'token_type_ids', 'attention_mask'],
        num_rows: 777
    })
})
```

**建立DataCollator**

```python
collator = DataCollatorWithPadding(tokenizer=tokenizer,return_tensors='pt')
collator

#==output==
DataCollatorWithPadding(tokenizer=BertTokenizerFast(name_or_path='google-bert/bert-base-chinese', vocab_size=21128, model_max_length=512, is_fast=True, padding_side='right', truncation_side='right', special_tokens={'unk_token': '[UNK]', 'sep_token': '[SEP]', 'pad_token': '[PAD]', 'cls_token': '[CLS]', 'mask_token': '[MASK]'}, clean_up_tokenization_spaces=False, added_tokens_decoder={
	0: AddedToken("[PAD]", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
	100: AddedToken("[UNK]", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
	101: AddedToken("[CLS]", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
	102: AddedToken("[SEP]", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
	103: AddedToken("[MASK]", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
}
), padding=True, max_length=None, pad_to_multiple_of=None, return_tensors='pt')
```


**建立DataLoader**

```python
from torch.utils.data import DataLoader
import torch

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-chinese")

trainset, validset = tokenizer_dataset['train'], tokenizer_dataset['test']
trainloader = DataLoader(trainset, batch_size=32, shuffle=True,collate_fn=collator) #collate_fn,建立處理batch內的資料
validloader = DataLoader(validset, batch_size=64, shuffle=False,collate_fn=collator)

next(enumerate(trainloader))[1]

#==output==
{'input_ids': tensor([[ 101, 4684, 2533,  ..., 3298, 7540,  102],
        [ 101, 1184, 5637,  ..., 1217,  802,  102],
        [ 101, 2769, 6221,  ...,    0,    0,    0],
        ...,
        [ 101,  127, 3299,  ..., 7279, 3300,  102],
        [ 101, 4692,  749,  ..., 8024, 2523,  102],
        [ 101, 7478, 2382,  ...,    0,    0,    0]]), 'token_type_ids': tensor([[0, 0, 0,  ..., 0, 0, 0],
        [0, 0, 0,  ..., 0, 0, 0],
        [0, 0, 0,  ..., 0, 0, 0],
        ...,
        [0, 0, 0,  ..., 0, 0, 0],
        [0, 0, 0,  ..., 0, 0, 0],
        [0, 0, 0,  ..., 0, 0, 0]]), 'attention_mask': tensor([[1, 1, 1,  ..., 1, 1, 1],
        [1, 1, 1,  ..., 1, 1, 1],
        [1, 1, 1,  ..., 0, 0, 0],
        ...,
        [1, 1, 1,  ..., 1, 1, 1],
        [1, 1, 1,  ..., 1, 1, 1],
        [1, 1, 1,  ..., 0, 0, 0]]), 'labels': tensor([1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1,
        0, 0, 1, 1, 1, 1, 1, 1])}
```


**建立模型和optimizer**

```python
from torch.optim import Adam
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained('google-bert/bert-base-chinese')
if torch.cuda.is_available():
    model = model.cuda()
optimizer = Adam(model.parameters(), lr=2e-5)
```

**建立訓練和評估的function**
- 本頁最後有說明train()和eval

```python
def evaluate()->float:
  model.eval() #讓模型進入評估模式
  validset_total = len(validset) #評估的筆數
  correct = 0 #預測正確的筆數
  for batch in validloader: #一次評估一個批次,目前每批次64筆
    if torch.cuda.is_available():
      batch = {k:v.cuda() for k, v in batch.items()}
    output = model(**batch)
    pred = torch.argmax(output.logits, dim=-1) #會有64個預測值
    correct += (pred.long() == batch['labels'].long()).float().sum() #每一批次正確的筆數
  acc = correct / validset_total #計算精準度
  return acc

def train(epoch=3, log_step=100):
  global_step = 0
  for ep in range(epoch):
    model.train() #讓模型進入訓練模式
    for batch in trainloader: #一次訓練一個批次,目前每批次32筆      
      if torch.cuda.is_available():
        batch = {k:v.cuda() for k, v in batch.items()}
      optimizer.zero_grad() #模型參數的梯度歸零
      output = model(**batch)
      output.loss.backward() #計算梯度
      optimizer.step() #更新模型參數
      if global_step % log_step == 0: #每100個批次,輸出一次損失梯度
        print(f"第{ep+1}躺,執行第{global_step}個批次,loss:{output.loss.item()}")
      global_step += 1 #每一批次就加1
    
    #每訓練一躺就評估一次精準度
    acc = evaluate()
    print(f"第{ep+1}躺,精準度:{acc}")
  

#==output==
第1躺,執行第0個批次,loss:0.09313313663005829
第1躺,執行第100個批次,loss:0.25313153862953186
第1躺,執行第200個批次,loss:0.28782710433006287
第1躺,精準度:0.9124839305877686
第2躺,執行第300個批次,loss:0.08671500533819199
第2躺,執行第400個批次,loss:0.10686539113521576
第2躺,精準度:0.9099099040031433
第3躺,執行第500個批次,loss:0.1386626660823822
第3躺,執行第600個批次,loss:0.02364831231534481
第3躺,精準度:0.9124839305877686
```

**訓練和評估模型**

```python
train()
```

**登入huggingface**

```python
from huggingface_hub import login

login()
```

**本地儲存**

```python
#儲存model,tokenizer,dataset於本地端
model.save_pretrained('./save_model') #save_model是資料夾名稱
tokenizer.save_pretrained('./save_model')
datasets.save_to_disk('./save_model')
```

**上傳至huggingface**

```python
#上傳model,tokenize和dataset
repo_id = "roberthsu2003/save_model"
model.push_to_hub(repo_id)
tokenizer.push_to_hub(repo_id)
datasets.push_to_hub(repo_id)
```

## **train(), eval()說明**

在 PyTorch 中，train() 主要用於將模型設置為「訓練模式」，但 不會執行實際的訓練過程。一般來說，完整的訓練過程需要包含以下幾個步驟：
1.	設定模型為訓練模式 (model.train())
2.	定義損失函數與優化器
3.	讀取訓練資料
4.	前向傳播 (Forward Pass)
5.	計算損失
6.	反向傳播 (Backward Pass)
7.	更新權重


### model.train() 的作用

model.train() 用來將模型切換為「訓練模式」，影響 某些特定層的行為，例如：
	•	Dropout：啟用隨機失活 (Dropout)，防止過擬合
	•	Batch Normalization：使用 mini-batch 內的統計數據 (均值、標準差) 來標準化輸入

當進行評估時，需要呼叫 model.eval()，讓模型進入測試模式。

2. PyTorch 訓練流程範例

```
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
```

# 1. 建立簡單模型
```
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 1)
        self.dropout = nn.Dropout(0.5)  # Dropout 層

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)  # 只有在 train() 模式下才會作用
        x = self.fc2(x)
        return x
```

# 2. 準備數據

```
x_train = torch.rand(100, 10)
y_train = torch.rand(100, 1)

dataset = TensorDataset(x_train, y_train)
dataloader = DataLoader(dataset, batch_size=10, shuffle=True)
```

# 3. 設定模型、損失函數、優化器
```
model = SimpleModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
```

# 4. 訓練模型
```
num_epochs = 10
for epoch in range(num_epochs):
    model.train()  # 設定為訓練模式

    for batch_x, batch_y in dataloader:
        optimizer.zero_grad()  # 清除梯度
        outputs = model(batch_x)  # 前向傳播
        loss = criterion(outputs, batch_y)  # 計算損失
        loss.backward()  # 反向傳播
        optimizer.step()  # 更新權重

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

print("訓練完成")
```

3. train() 和 eval() 的區別

```
模式	使用的函數	Dropout	Batch Normalization	反向傳播
訓練模式	model.train()	啟用	使用 mini-batch 統計數據	是
測試模式	model.eval()	停用	使用全局統計數據	否

在測試（評估）時，需確保模型進入 eval() 模式：

model.eval()  # 設定為測試模式
with torch.no_grad():  # 禁用梯度計算，節省記憶體與運算
    test_output = model(test_input)
```

4. 重要注意事項
- train() 不會執行訓練，它只是將模型設置為訓練模式。
- 訓練時，請確保：
- 調用 train()，確保 Dropout 和 BatchNorm 作用正確。
- 調用 optimizer.zero_grad()，清空梯度，避免影響下一次計算。
- 調用 loss.backward()，進行梯度反向傳播。
- 調用 optimizer.step()，更新權重。
```







