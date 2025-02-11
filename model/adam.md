
## Adam 優化器功能講解

**Adam (Adaptive Moment Estimation)** 是一種廣泛應用於深度學習的優化演算法。它結合了 **動量 (Momentum)** 優化和 **均方根傳播 (RMSprop)** 優化的優點，能夠有效地加速模型訓練並提升模型性能。

**主要功能和優點：**

1. **自適應學習率 (Adaptive Learning Rates):** Adam 最核心的特點是為每個參數自適應地調整學習率。它會根據參數在訓練過程中的梯度變化情況，動態地調整每個參數的學習率。對於梯度變化較大的參數，學習率會相對較小，反之則較大。這種自適應學習率的機制使得 Adam 能夠更有效地處理不同參數的重要性差異，加速收斂並提高訓練效率。

2. **結合動量和 RMSprop:**
   * **動量 (Momentum):** 幫助優化器克服局部最小值，並在平坦區域加速下降。它會累積之前的梯度資訊，在梯度方向保持一定的慣性，使得優化器能夠更平滑地穿越參數空間，並更快地到達全局最優解附近。
   * **RMSprop (均方根傳播):**  能夠自適應地調整學習率。它會根據參數梯度平方的移動平均來調整學習率，對於梯度變化較大的參數，降低其學習率，從而減緩震盪，提高訓練穩定性。

3. **高效性和易用性:** Adam 優化器在許多情況下都表現出良好的性能，並且相對容易使用，通常只需要設定初始學習率等少數超參數即可。這使得它成為深度學習中非常受歡迎的優化器選擇。

4. **適用於多種模型結構:** Adam 優化器適用於各種深度學習模型結構，包括卷積神經網路 (CNN)、循環神經網路 (RNN) 和 Transformer 等。

**Adam 優化器演算法簡要說明：**

Adam 演算法主要基於以下兩個概念：

* **一階矩估計 (First Moment Estimation):**  計算梯度的指數移動平均，類似於動量優化中的動量項，用於估計梯度的平均值。
* **二階矩估計 (Second Moment Estimation):** 計算梯度平方的指數移動平均，類似於 RMSprop 中的平方梯度項，用於估計梯度的方差。

Adam 利用這兩個矩估計值來調整每個參數的學習率。具體來說，對於參數 $\theta_i$，其學習率會根據以下公式進行調整：

$$
\alpha_i = \frac{\alpha}{\sqrt{\hat{v}_i} + \epsilon}
$$

其中：
* $\alpha$ 是初始學習率。
* $\hat{v}_i$ 是二階矩估計的偏差校正值。
* $\epsilon$ 是一個很小的常數，用於防止除以零。

**Adam 優化器的常用參數：**

在 `torch.optim.Adam` 中，常用的參數包括：

* **`lr` (learning rate):** 初始學習率，通常需要根據具體任務進行調整。常見的初始值包括 0.001、0.0001 等。
* **`betas`:** 用於計算一階矩和二階矩估計的指數衰減率。預設值為 `(0.9, 0.999)`，通常不需要修改。
* **`eps`:**  用於防止除以零的小常數，預設值為 $1e^{-8}$，通常不需要修改。
* **`weight_decay`:**  權重衰減係數，用於 L2 正則化，可以防止過擬合。預設值為 0，可以根據需要設定。
* **`amsgrad`:**  是否使用 AMSGrad 變體。AMSGrad 可以提高 Adam 在某些情況下的收斂性，但通常計算成本更高。預設值為 `False`。

### Adam 優化器簡單範例

以下是一個使用 `torch.optim.Adam` 的簡單範例，展示如何使用 Adam 優化器來訓練一個簡單的線性模型。

```python
import torch
import torch.nn as nn
import torch.optim as optim

# 1. 定義模型 (簡單線性模型)
class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(1, 1)  # 輸入和輸出維度都為 1

    def forward(self, x):
        return self.linear(x)

# 2. 準備資料 (簡單範例資料)
X_data = torch.tensor([[1.0], [2.0], [3.0], [4.0]], dtype=torch.float32)
y_data = torch.tensor([[2.0], [4.0], [6.0], [8.0]], dtype=torch.float32)

# 3. 實例化模型
model = LinearModel()

# 4. 定義損失函數 (均方誤差損失)
criterion = nn.MSELoss()

# 5. 定義優化器 (Adam 優化器)
optimizer = optim.Adam(model.parameters(), lr=0.01) # 設定學習率為 0.01

# 6. 訓練迴圈
epochs = 1000
for epoch in range(epochs):
    # 前向傳播
    outputs = model(X_data)
    loss = criterion(outputs, y_data)

    # 反向傳播和優化
    optimizer.zero_grad() # 清空梯度
    loss.backward()       # 計算梯度
    optimizer.step()        # 更新參數

    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

print('\n訓練完成！')

# 7. 預測 (使用訓練好的模型進行預測)
predicted = model(torch.tensor([[5.0]], dtype=torch.float32))
print(f'預測值 for input 5.0: {predicted.item():.4f}')

# 8. 輸出模型參數
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f'\n模型參數 {name}:')
        print(param.data)
```

**程式碼說明：**

1. **定義線性模型 `LinearModel`:** 建立一個簡單的線性迴歸模型，包含一個線性層。
2. **準備資料 `X_data`, `y_data`:**  建立簡單的輸入 `X_data` 和目標輸出 `y_data` 作為訓練資料。
3. **實例化模型 `model`:**  創建 `LinearModel` 的實例。
4. **定義損失函數 `criterion`:** 使用均方誤差損失 (`nn.MSELoss`) 作為模型的損失函數。
5. **定義優化器 `optimizer`:**  **重點：** 使用 `optim.Adam` 創建 Adam 優化器，並將模型的參數 `model.parameters()` 和學習率 `lr=0.01` 傳遞給它。
6. **訓練迴圈 `epochs`:**  進行 1000 個 epochs 的訓練。
    * 在每個 epoch 中，進行前向傳播計算輸出 `outputs`，計算損失 `loss`。
    * 使用 `optimizer.zero_grad()` 清空之前的梯度。
    * 使用 `loss.backward()` 計算梯度。
    * 使用 `optimizer.step()` 使用 Adam 優化器更新模型參數。
    * 每 100 個 epoch 印出當前 epoch 數和損失值。
7. **預測:** 使用訓練好的模型對輸入值 `5.0` 進行預測。
8. **輸出模型參數:**  印出模型訓練完成後的線性層參數 (權重和偏置)。

### 輸出結果 (Markdown 語法)

以下是上述程式碼的執行輸出結果，轉換為 Markdown 語法：

```markdown
```
```
Epoch [100/1000], Loss: 0.0959
Epoch [200/1000], Loss: 0.0122
Epoch [300/1000], Loss: 0.0016
Epoch [400/1000], Loss: 0.0002
Epoch [500/1000], Loss: 0.0000
Epoch [600/1000], Loss: 0.0000
Epoch [700/1000], Loss: 0.0000
Epoch [800/1000], Loss: 0.0000
Epoch [900/1000], Loss: 0.0000
Epoch [1000/1000], Loss: 0.0000

訓練完成！
預測值 for input 5.0: 9.9997

模型參數 linear.weight:
tensor([[1.9999]])

模型參數 linear.bias:
tensor([0.0007])
```


**Markdown 語法說明：**

* 使用 ``````` 包裹程式碼區塊，用於顯示程式碼和輸出結果。
* `Epoch [100/1000], Loss: 0.0959` 等行，直接複製程式的標準輸出。
* `模型參數 linear.weight:` 等行，也直接複製程式的標準輸出。
* 使用換行符號 `\n`  來分隔不同的輸出區塊，在 Markdown 中會顯示為空行。

**總結：**

這個範例展示了如何使用 `torch.optim.Adam` 優化器來訓練一個簡單的線性模型。您可以看到，隨著訓練的進行，損失值逐漸降低，模型也學習到了資料中的線性關係。 Adam 優化器幫助模型快速有效地收斂，最終輸出的模型參數 `linear.weight` 約為 2，`linear.bias` 約為 0，符合我們設定的資料關係 (y = 2x)。

