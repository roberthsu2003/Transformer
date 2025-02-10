# Model的基本使用範例
- 模型的下載和保存
	- 程式下載(download)和載入(load)
	- 模型下載(download)
	- 離線載入(load)
	- 模型配置的參數
- 模型的使用
	- 不帶model head的模型使用
	- 帶model head的模型使用

## 程式下載(download)和載入(load)

- 模型下載至本機(~/.cache)

```python
from transformers import AutoConfig, AutoModel, AutoTokenizer
model = AutoModel.from_pretrained('google-bert/bert-base-chinese')
```

## 模型下載至專案
- 使用git clone -> huggingface內有提供下載連結
- 必需先安裝git lfs(請查看官網)

```python
git lfs install
git clone https://huggingface.co/google-bert/bert-base-chinese
```

## 模型載入(本機)

```python
model = AutoModel.from_pretrained('./bert-base-chinese/')
```

## 模型配置的參數

**檢視模型配置的參數**

```python
model.config
```

**使用AutoConfig取得更多模型配置參數**

```python
config = AutoConfig.from_pretrained("google-bert/bert-base-chinese")
config
```

```python
#預設output_attentions預設為False
config.output_attentions
```


```
#可以檢查BertConfig的class程式碼,可以查到更多的參數
from transformers import BertConfig
```

## 使用`沒有model head`的模型,輸出的結果

**如何輸入token至模型**

```python
sen = "通訊等職類勞動力需求與職能基準研究"
model = AutoModel.from_pretrained('google-bert/bert-base-chinese')
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-chinese")
inputs = tokenizer(sen, return_tensors="pt") #使用pytorch的模型
inputs
```

```python
#輸出會看到last_hidden_state,還有hidden_states=None, past_key_values=None, attentions=None, cross_attentions=None

output = model(**inputs)
output
```

**要求輸出要有attentions的資料**

```
#由於有output_attentions=True,所以輸出有attentions的資料

model = AutoModel.from_pretrained('google-bert/bert-base-chinese',output_attentions=True)
sen = "通訊等職類勞動力需求與職能基準研究"
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-chinese") #output_attentions的是由配置查到的
inputs = tokenizer(sen, return_tensors="pt") #使用pytorch的模型
model(**inputs)
```

**inputs的input_id的長度**

```
len(inputs['input_ids'][0])

#==output==
19
```

**由於這個是沒有model head(沒有任務的model),所以模型輸出的是輸入token的張量**

```
output.last_hidden_state.size() #查詢輸出的維度

#==output==
torch.Size([1, 19, 768])

代表1個句子,有19個token,每個token有768張量
```














