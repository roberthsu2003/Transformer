## Model簡介
- 參考文章
> [Transformer — Attention Is All You Need](https://medium.com/ching-i/transformer-attention-is-all-you-need-c7967f38af14)
### Transformer的重要架構
1. The encoder-decoder framework
2. Attention mechanisms
3. Transfer learning

#### The encoder-decoder framework
- 比Transformer興起時就有的架構
在transformer之前，LSTM 等循環架構是 NLP 的最先進技術。這些架構在網路連線中包含一個回饋迴路，可讓資訊從一個步驟傳播到另一個步驟，因此非常適合用來建模文字等序列資料。

![](./images/pic1.png)

一個 RNN 接收一些輸入 (可能是單字或字元)，將其饋入網路，並輸出一個稱為隱藏狀態的向量(**hidden state**)。 與此同時，模型會透過回饋迴路將一些資訊回饋給它自己，然後在下一步中使用這些資訊。如圖所示：RNN 將每一步的狀態資訊傳遞到序列中的下一個操作。這些架構曾經（並將繼續）廣泛應用於 NLP 任務、語音處理和時間序列。RNN 發揮重要作用的一個領域是機器翻譯系統的開發，其目標是將一種語言的字詞序列映射到另一種語言。此類任務通常透過編碼器-解碼器或序列到序列架構，非常適合輸入和輸出都是任意長度序列的情況。編碼器的工作是將輸入序列的資訊編碼成一個數字表示，通常稱為最後的隱藏狀態。

![](./images/pic2.png)

如圖所示:一對 RNN 對此進行了說明，其中英語句子“Transformers are Great!”被編碼為隱藏狀態向量，然後被解碼以產生德語翻譯“Transformer sindgrosartig！”輸入字按順序饋送通過編碼器，並且從上到下一次產生一個輸出字。

儘管這種架構簡單優雅，但其弱點是編碼器的最終隱藏狀態造成了資訊瓶頸(_**information bottleneck**_)：它必須代表整個輸入序列的意義，因為這是解碼器在產生輸出時所能存取的全部資訊。這對於長序列來說尤其具有挑戰性，因為在將所有內容壓縮為單一固定表示法的過程中，序列開頭的資訊可能會遺失。

幸運的是，有一種方法可以擺脫這個瓶頸，即允許解碼器存取所有編碼器隱藏狀態。這一般的機制稱為注意力(_**attention**_)，是許多現代神經網路架構的關鍵元件。了解 RNN 的注意力是如何開發出來的，將有助於我們了解 Transformer 架構的主要構成元素之一。

#### Attention機制
**attention機縐**背後的主要想法是，編碼器不是將輸入序列資料產生單一個的隱藏狀態，而是在每一步驟輸出一個隱藏狀態，讓解碼器可以存取。然而，同時使用所有的狀態會對解碼器造成龐大的輸入，因此需要一些機制來決定使用狀態的優先順序。這就是注意力的作用：它讓解碼器在每個解碼時間步驟為每個編碼器狀態分配不同的權重，或稱「注意力」。

![](./images/pic3.png)

這個過程如圖所示，其中顯示了注意力在預測輸出序列中的第三個標記中的作用。透過在每個時間步驟中專注於哪些輸入標記是最相關的，這些基於注意力的模型能夠學習生成翻譯中的字詞與來源句子中的字詞之間的非三維對齊。

![](./images/pic4.png)

儘管注意力能夠產生更好的翻譯，但使用編碼器和解碼器的循環模型仍然存在一個主要缺點：計算本質上是順序的，不能在輸入序列上並行化。有了轉換器之後，我們就引進了一個新的建模範例：完全捨棄重複性，取而代之的是完全仰賴一種特殊形式的注意力，稱為自我注意（self-attention）。

![](./images/pic5.png)

基本想法是讓注意力作用於神經網路同一層的所有狀態。
其中編碼器和解碼器都有自己的自註意力機制，其輸出被饋送到前饋神經網路（FF NN）。這種架構的訓練速度比循環模型快得多並為 NLP 領域最近的許多突破鋪平了道路。在最初的 Transformer 論文中，翻譯模型是在各種語言的大型句子對語料庫上從頭開始訓練的。然而，在 NLP 的許多實際應用中，我們無法存取大量的標籤文字資料來訓練我們的模型。要開始變壓器革命，還差最後一塊：轉移學習。

#### NLP中的遷移學習

如今，電腦視覺領域的常見做法是使用遷移學習在一項任務上訓練 ResNet 等卷積神經網絡，然後使其適應或微調新任務。這使得網路能夠利用從原始任務中學到的知識。從架構上講，這涉及將模型分割成一個body和一個head，其中head是一個特定於任務的網路。在訓練期間，本體的權重會學習來源領域的廣泛特徵，這些權重會用來初始化新任務的新模型。

### HuggingFace中的Model Head

## Hugging Face 中的 Model Head 是什麼？

在 Hugging Face 的 `transformers` 庫中，**model head** 指的是負責特定任務（如分類、生成或回歸）的模型頂層部分。它通常是 Transformer 主體（backbone）之上的附加層，負責將 Transformer 提取的特徵轉換為具體的任務輸出。

---

## **Model Head 的概念**

Transformer 模型的架構可以拆分為兩個部分：

1. **Backbone（主體）**：
   - 例如 `BERT`, `GPT`, `T5` 這類 Transformer 模型的核心部分，負責語言理解或生成。
   - 它們的作用是學習語言的特徵表示（embeddings）。
   
2. **Head（頂層）**：
   - 根據任務需求，附加不同類型的 head，例如：
     - **分類（Classification Head）**：加上一個 `Linear` 層來進行文本分類。
     - **序列標註（Token Classification Head）**：用於命名實體識別（NER）。
     - **回歸（Regression Head）**：用來預測連續值，如情感分析中的分數。
     - **語言建模（LM Head）**：用於生成文本，如 GPT 模型中的 `LMHeadModel`。

---

## **常見的 Model Heads**

在 `transformers` 庫中，根據不同的 NLP 任務，Hugging Face 提供了一些內建的 model head：

| 任務 | Hugging Face 模型 |
|------|------------------|
| **文本分類**（Text Classification） | `BertForSequenceClassification`, `RobertaForSequenceClassification` |
| **序列標註**（Token Classification, NER） | `BertForTokenClassification`, `DistilBertForTokenClassification` |
| **問答**（Question Answering） | `BertForQuestionAnswering`, `AlbertForQuestionAnswering` |
| **文本生成**（Text Generation） | `GPT2LMHeadModel`, `T5ForConditionalGeneration` |
| **翻譯**（Translation） | `MarianMTModel`, `T5ForConditionalGeneration` |
| **摘要**（Summarization） | `BartForConditionalGeneration`, `T5ForConditionalGeneration` |

---

## **Model Head 的運作方式**

這些 model head 本質上是在 Transformer backbone 之上加了一些額外的神經網絡層。例如：

### 1. **分類任務 (`BertForSequenceClassification`)**
- 在 BERT 的輸出 `[CLS]` token 上加一個 `Linear` 層，並使用 `Softmax` 來進行分類。

```python
from transformers import BertForSequenceClassification

model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)  # 二分類
```

### 2. **語言模型 (`GPT2LMHeadModel`)**
- GPT-2 生成模型的 `head` 是一個線性層，將隱藏狀態映射到詞彙表，然後用 `Softmax` 來產生下一個 token。

```python
from transformers import GPT2LMHeadModel

model = GPT2LMHeadModel.from_pretrained("gpt2")
```

### 3. **問答 (`BertForQuestionAnswering`)**
- 這個 head 會預測答案在文本中的起始位置和結束位置。

```python
from transformers import BertForQuestionAnswering

model = BertForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
```

---

## **如何自定義 Model Head**

你可以自定義一個 model head，以下是一個簡單的例子：

```python
import torch
from torch import nn
from transformers import BertModel

class CustomBERTClassifier(nn.Module):
    def __init__(self, model_name="bert-base-uncased", num_classes=2):
        super(CustomBERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # 取 [CLS] token
        logits = self.classifier(cls_output)
        return logits
```

這樣，你就可以根據需求設計自己的 head 來做不同的 NLP 任務。

---

## **結論**

Hugging Face 的 `transformers` 提供了多種 `model heads`，適用於不同的 NLP 任務。這些 `heads` 只是 Transformer backbone 之上的一層或多層神經網絡，它們能夠將通用的語言表徵轉換為具體任務的輸出。如果你有更特殊的需求，也可以自定義自己的 `model head`。

如果你需要針對特定應用調整 `model head`，可以進一步使用 **Fine-tuning** 來適配特定數據集。🚀








