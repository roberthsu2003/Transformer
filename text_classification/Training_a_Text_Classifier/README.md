# 訓練文字分類
(Training a Text Classifier)

DistilBERT 等模型經過預先訓練，可以預測文字序列中的遮罩字。但是，我們不能直接使用這些語言模型進行文字分類；我們需要對它們稍加修改。為了了解需要進行哪些修改，讓我們來看看 DistilBERT 等基於編碼器的模型的架構。

![](./images/pic1.png)

文字被標記化並表示為獨熱向量（稱為標記編碼）, 標記器詞彙表的大小決定了標記編碼的維度，它通常由 20k-200k 個唯一標記組成。接下來，這些標記編碼被轉換為標記嵌入，即存在於低維空間中的向量。然後，標記嵌入透過編碼器區塊層來產生每個輸入標記的隱藏狀態。對於語言建模的預訓練目標，每個隱藏狀態都被饋送到預測遮罩輸入標記的層。對於分類任務，我們用分類層取代語言建模層。

**2種方式訓練此模型**

**1.特徵提取(Feature extraction)**

我們使用隱藏狀態作為特徵並在其上訓練分類器，而不修改預訓練模型。

**2.微調(Fine-tuning)**

我們對整個模型進行端到端的訓練，同時也更新了預訓練模型的參數。

## 使用特微提取
使用transformer作為特徵提取器相當簡單,如下圖所示。我們在訓練過程中凍結身體的體重，並使用隱藏狀態作為分類器。

![](./images/pic2.png)

### 使用預訓練模型
我們將使用 Transformers 中另一個方便的自動類，名為 AutoModel。讓我們使用此方法來載入DistilBERT checkpoint：

```python
from transformers import AutoModel
import torch

model_ckpt = 'distilbert-base-uncased'
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
model = AutoModel.from_pretrained(model_ckpt).to(device)
```

**提取最後的隱藏狀態(Extracting the last hidden states)**

這裡我們使用 PyTorch 來檢查 GPU 是否可用，然後將 PyTorch nn.Module.to() 方法連結到模型載入器。這確保了模型將在 GPU 上運行（如果我們有的話）。如果不是，模型將在 CPU 上運行，這會慢得多。

AutoModel 類別將標記編碼轉換為嵌入，然後將它們輸入編碼器堆疊以傳回隱藏狀態。

為了熱身，讓我們檢索單一字串的最後隱藏狀態。我們需要做的第一件事是對字串進行編碼，並將標記轉換為 PyTorch 張量。這可以透過向標記器提供 return_tensors="pt" 參數來實現，如下所示：

```python
from transformers import AutoTokenizer
model_ckpt = 'distilbert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
```

```
text = 'this is a test'
inputs = tokenizer(text, return_tensors="pt")
print(f"Input tensor shape:{inputs['input_ids'].size()}")

#==output==
Input tensor shape:torch.Size([1, 6])

```

我們可以看到，得到的張量形狀為 [batch_size, n_tokens]. 現在我們已經將編碼作為張量，最後一步是將它們放在與模型相同的裝置上並傳遞輸入，如下所示：

```
inputs = {k:v.to(device)for k, v in inputs.items()}
inputs

#==output==
{'input_ids': tensor([[ 101, 2023, 2003, 1037, 3231,  102]]),
 'attention_mask': tensor([[1, 1, 1, 1, 1, 1]])}
```

```
with torch.no_grad():
    outputs = model(**inputs)
print(outputs)

#==output==
BaseModelOutput(last_hidden_state=tensor([[[-0.1565, -0.1862,  0.0528,  ..., -0.1188,  0.0662,  0.5470],
         [-0.3575, -0.6484, -0.0618,  ..., -0.3040,  0.3508,  0.5221],
         [-0.2772, -0.4459,  0.1818,  ..., -0.0948, -0.0076,  0.9958],
         [-0.2841, -0.3917,  0.3753,  ..., -0.2151, -0.1173,  1.0526],
         [ 0.2661, -0.5094, -0.3180,  ..., -0.4203,  0.0144, -0.2149],
         [ 0.9441,  0.0112, -0.4714,  ...,  0.1439, -0.7288, -0.1619]]]), hidden_states=None, attentions=None)
```

這裡我們使用了 torch.no_grad() 上下文管理器來停用梯度的自動計算。這對於推理很有用，因為它減少了記憶體佔用。列印計算結果。根據模型配置，輸出可以包含多個對象，例如隱藏狀態、損失或註意力，它們排列在類似於 Python 中的命名元組的類別中。
在我們的範例中，模型輸出是 BaseModelOutput 的一個實例，我們可以簡單地透過名稱存取其屬性。當前模型僅傳回一個屬性，即最後一個隱藏狀態，因此讓我們檢查一下它的形狀：

```
outputs.last_hidden_state.size()

#==output==
torch.Size([1, 6, 768])
```

看看隱藏狀態張量，我們發現它的形狀為 [batch_size,n_tokens, hidden_​​dim]。換句話說，傳回一個 768 維向量6個輸入標記中的每一個。對於分類任務，通常做法是只使用與 [CLS] 標記相關的隱藏狀態作為輸入特徵。

```
outputs.last_hidden_state[:,0].size()

#==output==
torch.Size([1, 768])
```

我們知道如何獲取單一字串的最後一個隱藏狀態, 讓我們透過建立一個儲存所有這些向量的新 hidden_​​state 列來對整個資料集執行相同的操作。正如我們對標記器所做的那樣，我們將使用 DatasetDict 的 map() 方法一次提取所有隱藏狀態。我們需要做的第一件事是將前面的步驟包裝在一個處理函數中：

```python
from datasets import load_dataset
from transformers import AutoTokenizer
emotions = load_dataset("emotion")
model_ckpt = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
def tokenize(batch):
    return tokenizer(batch['text'], padding=True, truncation=True)

def extract_hidden_states(batch):
    inputs = {k:v.to(device) for k,v in batch.items() if k in tokenizer.model_input_names}
    with torch.no_grad():
        last_hidden_state = model(**inputs).last_hidden_state
    return {"hidden_state": last_hidden_state[:,0].cpu().numpy()}

emotions_encoded = emotions.map(tokenize, batched=True, batch_size=None)
emotions_encoded.set_format("torch", columns=["input_ids", "attention_mask", "label"])
emotions_hidden = emotions_encoded.map(extract_hidden_states, batched=True)
emotions_hidden["train"].column_names

#==output==
['attention_mask', 'hidden_state', 'input_ids', 'label', 'text']
```

```
import numpy as np

X_train = np.array(emotions_hidden['train']['hidden_state'])
X_valid = np.array(emotions_hidden['validation']['hidden_state'])
y_train = np.array(emotions_hidden['train']['label'])
y_valid = np.array(emotions_hidden['validation']['label'])

X_train.shape, X_valid.shape
```

未結束



















