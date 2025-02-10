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

## 清理數據

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

## 建立pytorch的Dataset
[**pytorch DataSet的簡單範例**](./pytorch_dataset.md)

```
```