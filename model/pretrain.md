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

trainloader = DataLoader(trainset, batch_size=5, shuffle=True)
validloader = DataLoader(validset, batch_size=64, shuffle=True)

next(enumerate(trainloader))[1]

#==output==
[('錦江貴賓樓房間裝修還可以，環境一般，我認為本次入住餐飲服務比較差，比如我點菜時他告訴我68元，但結帳時時98元。而且價格比較高。',
  '老酒店翻新，總體來說還可以，價效比比較高',
  '酒店實在太差了,房間隔音極差,北塔挨著迪吧，快曲放到半夜3點；換到南塔，又挨著KTV音樂響到半夜2點，根本睡不著覺！換房間的時候服務員態度又很差，真是從來沒住過這麼差的酒店',
  '房間還是不錯的，住的楓丹白露，房間很大，比較有特色，有比較齊全的廚房裝置，如果一家人出去玩可以買一些簡單的食物自己烹調。不過服務要差一些，我們住了兩天居然沒有人來打掃房間，住了很多家酒店只有這家是這樣的。後來問了前臺，前臺答覆這裡主要是會所，不太有經驗。總體來說態度還是不錯的。早餐比較差，除了包子就是花捲，兩天都是這樣，一碗白粥、一杯牛奶，因此三天的正餐都不敢在這裡吃。還好離縣城不遠可以選擇到縣城吃，或在路口吃農家飯。不過出去玩住著還不錯，很安靜，周圍都是別墅，門口有保安，相對安全，走出去不遠就是紅螺湖，我們沿著山路開車上去，走到有水的地方就停下休息，玩的很好。懷柔人民越來越會做生意，凡是能停車的水邊全都有人“把守”，需要按人頭收費才能玩，每人5元，價錢不高，但總讓人有點感慨。總的來說五一還是很愉快。酒店人不多，下次可能還會選擇。',
  '五星級酒店空降在富陽這個小地方。硬體沒有什麼可抱怨的，大堂氣派非凡，杭州也很難挑出可以比的。江景一覽無餘。除了房間的一些小細節做的和大城市的五星級還有些小差距，但是在富陽，你就別想找個更好的地方了。這個酒店要是在杭州，就是800元一晚也找不到，500多的房價算是可以了。問題是，問什麼在富陽會有這麼好的酒店？'),
 tensor([0, 1, 0, 1, 1])]```





