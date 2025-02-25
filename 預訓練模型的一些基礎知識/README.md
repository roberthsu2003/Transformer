
# 預訓練模型（Pre-trained Model）介紹

## 什麼是預訓練模型？
預訓練模型指的是已經在大型數據集上訓練過的機器學習或深度學習模型，這些模型已經學習了豐富的特徵與模式，開發者可以直接使用它們來進行特定任務，而不需要從零開始訓練。

這種方法大幅節省了計算資源與時間，並且透過**遷移學習（Transfer Learning）**，我們可以在較小的數據集上進行**微調（Fine-tuning）**，讓預訓練模型適應特定應用。

## 預訓練模型的特點

- **已學習到通用的特徵**：例如影像識別模型已經學習到邊緣、形狀、顏色等資訊。
- **大幅減少訓練時間**：不需要重新訓練整個深度學習模型。
- **提高模型效能**：由於模型已在大規模數據上訓練，通常能提供較好的準確率。
- **適用於多種領域**：包括電腦視覺（CV）、自然語言處理（NLP）、語音識別等。

---

## 常見的預訓練模型

### 1. 電腦視覺（Computer Vision, CV）

| **模型名稱** | **簡介** | **主要應用領域** |
|-------------|---------|----------------|
| ResNet | 深層殘差網絡，適合圖像分類。 | 影像分類、特徵提取 |
| VGG16/VGG19 | 深層卷積神經網絡（CNN），適用於影像識別。 | 物件偵測 |
| Inception (GoogleNet) | 多尺度特徵學習，提高識別能力。 | 影像分類 |
| EfficientNet | 高效能 CNN，運算速度快且準確度高。 | 影像分類 |
| YOLO (You Only Look Once) | 即時物件偵測模型，速度快。 | 物件偵測 |
| Mask R-CNN | 偵測物件並進行像素級分割。 | 影像分割 |

這些模型通常可以在 **PyTorch** 的 `torchvision.models` 或 **TensorFlow** 的 `tf.keras.applications` 直接載入使用。

### 2. 自然語言處理（Natural Language Processing, NLP）

| **模型名稱** | **簡介** | **主要應用領域** |
|-------------|---------|----------------|
| BERT (Bidirectional Encoder Representations from Transformers) | 雙向 Transformer，能理解上下文語意。 | 文字分類、QA、翻譯 |
| GPT (Generative Pre-trained Transformer) | 生成式語言模型，適合對話生成、文章創作。 | 自然語言生成 (NLG) |
| T5 (Text-to-Text Transfer Transformer) | 將所有 NLP 任務轉換為文字生成問題。 | 文字摘要、機器翻譯 |
| XLNet | 改進 BERT，增加了預測能力。 | NLP 分類、摘要 |
| RoBERTa | 改進 BERT，增強語言理解能力。 | 文字理解 |

這些 NLP 模型可以在 **Hugging Face Transformers** 庫中找到，直接下載並應用。

### 3. 語音識別（Speech Recognition）

| **模型名稱** | **簡介** | **主要應用領域** |
|-------------|---------|----------------|
| DeepSpeech | 由 Mozilla 開發的開源語音轉文字（ASR）模型。 | 語音識別 |
| wav2vec 2.0 | Facebook AI 開發，能夠無監督學習語音特徵。 | 語音轉文字 |
| Whisper (OpenAI) | OpenAI 開發的多語言語音辨識模型。 | 語音識別、字幕生成 |

這些模型可以透過 **PyTorch (`torchaudio`)** 或 **Hugging Face** 直接下載使用。

---

## 如何使用預訓練模型？

### 1. 在電腦視覺中使用 ResNet

```python
import torch
import torchvision.models as models

# 載入 ResNet50 預訓練模型
model = models.resnet50(pretrained=True)
model.eval()  # 設定為評估模式

# 進行推論
# model.forward(input_tensor)
```


**2. 在 NLP 中使用 BERT**

```other
from transformers import BertTokenizer, BertModel

# 載入 BERT 預訓練模型
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

# 文字 Tokenize
text = "Hello, how are you?"
inputs = tokenizer(text, return_tensors="pt")

# 取得輸出向量
outputs = model(**inputs)
```


**預訓練模型的應用場景**

| **應用場景** | **適用模型**             |
| -------- | -------------------- |
| 影像分類     | ResNet, EfficientNet |
| 物件偵測     | YOLO, Faster R-CNN   |
| 語音轉文字    | Whisper, DeepSpeech  |
| 自動對話系統   | GPT, ChatGPT         |
| 機器翻譯     | T5, MarianMT         |
| 情感分析     | BERT, RoBERTa        |

**預訓練模型 vs. 從零開始訓練**

| **方式**      | **優點**          | **缺點**       |
| ----------- | --------------- | ------------ |
| **使用預訓練模型** | 訓練時間短、資源消耗少、效果好 | 可能無法完全符合特定需求 |
| **從零開始訓練**  | 完全客製化、靈活調整      | 訓練成本高，需要大量數據 |

對於大部分應用，使用**預訓練模型 + 微調 (Fine-tuning)** 是最佳選擇，既能保持高準確度，又能節省資源。

**總結**

• **預訓練模型**是已經在大規模數據上訓練好的模型，可直接使用或進行微調。

• **廣泛應用於**：電腦視覺 (CV)、自然語言處理 (NLP)、語音識別 (Speech)。

• **可以透過** TensorFlow、PyTorch 或 Hugging Face **輕鬆載入**。

• **適用於**影像分類、物件偵測、機器翻譯、對話系統等多種 AI 應用。

