**1. accuracy_score 是什麼？**

accuracy_score 用來計算分類模型的 **準確率（Accuracy）**，其公式如下：

它衡量的是模型預測正確的比例，適用於**類別分佈均衡**的數據集。例如，如果模型預測 100 個樣本，其中 90 個正確，那麼準確率就是 90%。

📌 適**用場景：**

• 當數據集的類別分佈較為均衡時，accuracy_score 是一個可靠的評估指標。

📌 使**用範例：**

```other
from sklearn.metrics import accuracy_score

y_true = [0, 1, 1, 1, 0, 1, 0, 0, 1, 0]  # 真實標籤
y_pred = [0, 1, 1, 0, 0, 1, 0, 1, 1, 0]  # 預測標籤

acc = accuracy_score(y_true, y_pred)
print("Accuracy:", acc)
```


**2. f1_score 是什麼？**

f1_score 是 **F1 分數**，用來綜合評估 **Precision（精確率）** 和 **Recall（召回率）**，其公式如下：

其中：

• **Precision（精確率）**：模型預測為正類的樣本中，實際為正類的比例。

• **Recall（召回率）**：實際為正類的樣本中，被模型正確識別的比例。

📌 適**用場景：**

• 當數據集 **不均衡**（例如正類樣本遠少於負類樣本）時，accuracy_score 可能會導致誤導性的結果，而 f1_score 能更好地反映模型性能。

📌 使**用範例：**

```other
from sklearn.metrics import f1_score

y_true = [0, 1, 1, 1, 0, 1, 0, 0, 1, 0]
y_pred = [0, 1, 1, 0, 0, 1, 0, 1, 1, 0]

f1 = f1_score(y_true, y_pred)
print("F1 Score:", f1)
```


**3. accuracy_score vs f1_score 差異：**

| **指標**         | **適用場景**     |
| -------------- | ------------ |
| accuracy_score | 適用於類別均衡的數據集  |
| f1_score       | 適用於類別不均衡的數據集 |

如果數據集中某個類別佔比極低（例如 95% 是類別 A，5% 是類別 B），accuracy_score 可能會給出 95% 的高準確率，但模型實際上完全無法識別類別 B。這時候 f1_score 會是更好的評估標準。

🚀 總**結：**

• 如果類別均衡，使用 **accuracy_score**。

• 如果類別不均衡，使用 **f1_score**。
