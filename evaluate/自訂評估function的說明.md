```python
import evaluate
acc_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

def eval_metric(eval_predict):
    predictions, labels = eval_predict
    predictions = predictions.argmax(axis=-1)
    acc = acc_metric.compute(predictions=predictions, references=labels)
    f1 = f1_metric.compute(predictions=predictions, references=labels)
    acc.update(f1)
    return acc

```

這段程式碼的主要目的是在 **Transformer 的文本分類任務** 中評估模型的性能，使用 **準確率（Accuracy）** 和 **F1 分數（F1-score）** 兩個指標。讓我們逐行分析其功能與設計原因。

**程式碼解析**

**1. 匯入 evaluate 函式庫**

```other
import evaluate
```


• evaluate 是 Hugging Face 提供的 **指標評估庫**，用來計算機器學習任務的標準指標，例如 **accuracy**、**F1-score**、**precision**、**recall** 等。

• 它的優勢是 **簡單易用**，可以直接載入不同的評估指標，而不需要自己實作。

**2. 載入 accuracy 和 f1 兩種評估指標**

```other
acc_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")
```


• evaluate.load("accuracy")：載入 **準確率（Accuracy）** 評估函式，這是分類任務中最常見的指標，計算公式如下：

• evaluate.load("f1")：載入 **F1-score**，這個指標是 **Precision（精確率）** 和 **Recall（召回率）** 的調和平均數：

• **Precision（精確率）**：TP / (TP + FP)

• **Recall（召回率）**：TP / (TP + FN)

• F1-score **適合於非均衡數據集**，因為它考慮了 **錯誤分類的影響**。

**3. 定義 eval_metric 評估函數**

```other
def eval_metric(eval_predict):
```


• 這是一個接收 eval_predict 參數的函式，通常在 Transformer 訓練或評估過程中 **由 Trainer 提供**。

• eval_predict 來自 Trainer.evaluate()，通常是一個 **(預測結果, 真實標籤)** 的元組。

**4. 解析 eval_predict，取得預測與真實標籤**

```other
predictions, labels = eval_predict
```


• eval_predict 是一個元組 (predictions, labels)：

• predictions：模型的輸出，通常是 **logits（未經 softmax 的分數）**，形狀為 (batch_size, num_classes)。

• labels：對應的 **真實標籤**，形狀為 (batch_size,)。

**5. 將 logits 轉換為最終預測標籤**

```other
predictions = predictions.argmax(axis=-1)
```


• 由於 Transformer 模型的輸出是 logits，需要將其轉換為類別標籤：

• argmax(axis=-1)：在 **最後一個維度（num_classes）** 上選擇最大值的索引，即最可能的類別。

• 例如：

```other
predictions = [[2.3, 5.1, 1.8],  # logit 分數
               [1.2, 3.4, 6.7]]
predictions.argmax(axis=-1)  # 選出最大值的索引
# Output: [1, 2]  (代表預測類別)
```


**6. 計算 accuracy**

```other
acc = acc_metric.compute(predictions=predictions, references=labels)
```


• acc_metric.compute() 會計算準確率：

• predictions=predictions：模型的預測結果（類別標籤）。

• references=labels：對應的真實標籤。

• **範例：**

```other
acc_metric.compute(predictions=[1, 0, 1], references=[1, 1, 1])
# Output: {'accuracy': 0.6667}  (因為 2/3 個預測是正確的)
```


**7. 計算 F1-score**

```other
f1 = f1_metric.compute(predictions=predictions, references=labels)
```


• f1_metric.compute() 會計算 F1-score：

• **適合多分類（multi-class classification）或不平衡數據集（imbalanced dataset）**。

• 例如：

```other
f1_metric.compute(predictions=[1, 0, 1], references=[1, 1, 1])
# Output: {'f1': 0.8}  (假設 precision 和 recall 分數計算後)
```


**8. 更新 acc 字典，合併 f1 分數**

```other
acc.update(f1)
```


• update() 會把 f1 計算出的字典加入 acc：

• acc 原本是：{'accuracy': 0.6667}

• f1 可能是：{'f1': 0.8}

• 合併後變成：

```other
{'accuracy': 0.6667, 'f1': 0.8}
```


**9. 回傳綜合評估結果**

```other
return acc
```


• eval_metric() 會回傳一個 **包含準確率與 F1-score 的字典**。

• 範例輸出：

```other
{'accuracy': 0.75, 'f1': 0.82}
```


• **此結果可以用來監測 Transformer 模型的學習進度**。

**為何這樣設計？**


1. **簡潔且模組化**

• evaluate 函式庫簡化了指標計算，不需要手動計算 Accuracy 和 F1-score。

• argmax() 讓 logits 轉換為類別標籤，適用於大多數分類模型。


1. **支援 Trainer**

• Hugging Face Trainer 允許使用 compute_metrics 來指定評估函數，這樣可以自動在訓練和評估時計算 Accuracy 和 F1-score。


1. **適合多分類問題**

• 既能處理 **二元分類**，也能適用於 **多分類問題**，因為 F1-score 會根據 num_classes 自動計算加權平均。


1. **可擴展性**

• eval_metric() 可以擴展成 **計算更多指標**（如 Precision、Recall）：

```other
precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")
```


**結論**

這段程式碼適用於 Transformer **文本分類（text classification）** 模型，能夠計算 **準確率（Accuracy）** 和 **F1-score**，並整合成一個回傳值，方便在 **Hugging Face Trainer 訓練流程中自動評估**。

**補充：如何在 Trainer 使用這個函式？**

如果使用 Trainer 進行 Transformer 訓練，可以這樣傳遞 eval_metric()：

```other
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=eval_metric  # 這裡指定我們的函數
)
```


這樣 Trainer.evaluate() 會自動計算 Accuracy 和 F1-score，無需手動調用。
