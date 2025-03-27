# HuggingFace TransFormers
## [預訓練模型的介紹](./預訓練模型的一些基礎知識)
## 1. HuggingFace提供的主要類別
### 1.1 [基礎知識和環境安裝](./環境安裝)
### 1.2 [pipeline連結管線](./pipeline/)
### 1.3 [tokenizer分詞器](./tokenizer/)
### 1.4 [Model模型說明](./model/)
- [**Model 模型使用**](./model/example.md)
- [**Model 模型預訓練**](./model/pretrain.md)
### 1.5 [Datasets資料集](./datasets/)
### 1.6 [Evaluate評估](./evaluate/)
- [自訂評估function的說明](./evaluate/自訂評估function的說明.md)
### 1.7 [trainer訓練器](./trainer/)
### 1.8 [基於Transformers的NL解決方案](./基於Transformers的NL解決方案)
### 1.9 [動態調整訓練時記憶體使用](./動態調整訓練時記憶體使用)

## 2. 依任務類別的微調(Fine Tune)
### [自然語言Transformer的種類](./實戰運用/選擇適合的預訓練模型)
- 選擇適合的預訓練模型

### [文本分類(Text Classification)](./實戰運用/text_classification)

### 命名實體識別(Named Entity Recognition,NER)
- [實務應用說明](./實戰運用/NER/實務運用說明.md)
- [資料標識說明](./實戰運用/NER/資料標示說明.md)
- [評估指標說明](./實戰運用/NER/評估指標說明.md)
- [建議使用的基礎模型](./實戰運用/NER/建議使用的基礎模型.md)
- [實作](./實戰運用/NER)

### 機器閱讀理解(Question Answering)
- [機器閱讀理解](./實戰運用/QuestionAnswering/什麼是機器閱讀理解.md)

- [機器閱讀理解應用情境](./實戰運用/QuestionAnswering/機器閱讀理解應用情境.md)

- [資料集](./實戰運用/QuestionAnswering/資料集說明.md)

	- [下載台達研究院QA專用資料資](https://github.com/DRCKnowledgeTeam/DRCD)  
		1. DRCD_dev.json  
		2. DRCD_test.json  
		3. DRCD_training.json

	- [將DRCD資料轉換為cmrc2018格式實作ipynb](./實戰運用/QuestionAnswering/DRCD資料集轉換為CMRC2018/將DRCD資料轉換為cmrc2018格式.ipynb)  
		1. 轉換為CMRC2018格式
		2. 上傳至HuggingFace DataSet

- [截斷策略訓練實作](./實戰運用/QuestionAnswering/載斷策略實作)

- [滑動視窗策略訓練實作](./實戰運用/QuestionAnswering/滑動策略實作)

- [下載截斷策略模型實作](./實戰運用/QuestionAnswering/載斷策略實作/下載模型實作.ipynb)














