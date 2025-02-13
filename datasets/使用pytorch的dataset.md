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

```python
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
('距離川沙公路較近,但是公交指示不對,如果是"蔡陸線"的話,會非常麻煩.建議用別的路線.房間較為簡單.', np.int64(1))
```

### 步驟3-分組(切割)Dataset,成為trainset和validset2組
- random_split() - 切割和重新隨機排序
- 傳出Subset類別

```python
from torch.utils.data import random_split

trainset, validset= random_split(hotelDataSet,lengths=[0.9,0.1])
trainset[0], len(trainset), len(validset)

#==output==
(('因為公司的報銷制度嚴格,所以我在酒店結帳的時候讓前臺在列印出來的明細單上加蓋酒店的章(大家說這個要求過分嗎),但是前臺小姐以各種理由拒不蓋章,試問如果出差回去不能報銷費用的話,這樣的酒店誰還敢住?如果不是急著趕飛機的話,一定要找酒店要個說法!建議報銷制度嚴格的朋友不要考慮這個酒店了.',
  np.int64(0)),
 6989,
 776)
```