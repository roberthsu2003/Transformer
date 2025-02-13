## 使用pytorch的Dataset
### 最終目標訓練的資料集格式
- dictionary
	- input_ids:tensor
	- token_type_ids:tensor
	- attention_mask:tensor
	- labels:tensor

| input_ids | toke_type_ids | attention_mast | labels |
|:--|:--|:--|:--|
| tensor | tensor | tensor | tensor |

### 步驟1-資料轉換和整理
- 使用pandas
- info(),dropna(),len()

```python
import pandas as pd

data = pd.read_csv('ChnSentiCorp_htl_all.csv')
data.head()
```

![](./images/pic1.png)

---

```python
data = data.dropna()
len(data)

#==output==
7765
```

### 步驟2-將資料轉換成為DataSet格式
- 必需自訂類別繼承Dataset類別
- 建立實體有iterable和subscript的能力(實體[index])
- 實作__len__
- 實作__getitem__,傳出tuple資料

```
from torch.utils.data import Dataset

class HotelDataSet(Dataset):
    def __init__(self):
        super().__init__()
        data = pd.read_csv('ChnSentiCorp_htl_all.csv')
        self.data = data.dropna()

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index): #傳出tuple
        return self.data.iloc[index]['review'], self.data.iloc[index]['label']

hotelDataSet = HotelDataSet()
hotelDataSet[0]

#==output==
```