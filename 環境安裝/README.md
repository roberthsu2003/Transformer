# Transformer 基礎知識和環境安裝

## 常見自然語言處理任務
- **情緒分析(sentiment-analysis)**
- **文本生成(text-generation)**
- **命名實體識別(Named Entity Recognition)**
- **閱讀理解(question-answering)**

給予一編相關文章和問題,從文章內取得答案

- **填充答案(fill-mask)**

填充題目的回答

- **內容摘要(summarization)**
- **翻譯(translation)**
- **特徵提取(feature-extraction)**
- **對話機器人(conversation)**

## Transformer簡單介紹
- HuggingFace出品,常用的自然語言處理套件之一
- HuggingFace提供了大量基於Transformer架構的預訓練模型,除自然語言外還有圖像,音頻和多模態的模型
- 良好的分數和提供大量數據集,也支援使用者上傳
- 快整實現預訓練模型的fine tune

## Transformers相關的套件包
- **Transformers**
	- 核心套件
	- 模型加載
	- 模型訓練
	- 流水線

- **Tokenizer**
	- 分詞器
	- 對數據進行預處理
	- 內容到token序列互相轉換

- **Datasets**
	- 資料集,提供資料集的下載和處理

- **Evaluate**
	- 評估功能,提供各種評價指標的計算函式

- **PEFT**
	- 高效微調模型,提供幾種高效能微調方法

- **Accelerate**
	- 分布式訓練

- **Optimum**
	- 最佳化和加速,支援Onnxruntime, OpenVino

- **Gradio**
- 視覺化介面部署


## 環境安裝
1. **重要的安裝**
- pytouch
- git
- github帳號
- hugging face帳號
- vscode
	- jupyter
	- python
- miniconda

2. 套件安裝(conda install)

- transformers
- datasets
- evaluate
- peft
- accelerate
- gradio==4.44.1(範例才可以執行,5.x版以上要python3.10)
- optimum
- sentencepiece
- jupyterlab
- scikit-learn
- pandas
- matplotlib
- tensorboard
- nltk
- rouge
