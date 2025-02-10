# 自定義 PyTorch Dataset 範例

在 PyTorch 中，可以透過繼承 `torch.utils.data.Dataset` 來建立自定義的 Dataset。  
以下是一個簡單的範例，示範如何實作 `__len__` 和 `__getitem__` 方法。

## 1. 匯入必要的函式庫

```python
import torch
from torch.utils.data import Dataset

2. 自訂 Dataset 類別

class CustomDataset(Dataset):
    def __init__(self, data, labels):
        """
        初始化 Dataset，儲存資料和標籤。
        :param data: list or numpy array，特徵數據
        :param labels: list or numpy array，對應的標籤
        """
        self.data = data
        self.labels = labels

    def __len__(self):
        """ 返回數據集的大小 """
        return len(self.data)

    def __getitem__(self, index):
        """
        根據索引獲取單個樣本
        :param index: int，索引值
        :return: (data, label) tuple
        """
        x = torch.tensor(self.data[index], dtype=torch.float32)
        y = torch.tensor(self.labels[index], dtype=torch.long)
        return x, y

3. 測試自定義 Dataset

# 假設有一些簡單的數據
data = [[1, 2], [3, 4], [5, 6], [7, 8]]
labels = [0, 1, 0, 1]

# 建立 Dataset 物件
dataset = CustomDataset(data, labels)

# 測試 Dataset
print(f"Dataset 大小: {len(dataset)}")
for i in range(len(dataset)):
    x, y = dataset[i]
    print(f"索引 {i}: x={x}, y={y}")

4. 與 DataLoader 搭配使用

from torch.utils.data import DataLoader

# 建立 DataLoader
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# 迭代 DataLoader
for batch_idx, (x_batch, y_batch) in enumerate(dataloader):
    print(f"Batch {batch_idx}:")
    print(f"x_batch: {x_batch}")
    print(f"y_batch: {y_batch}")

這段程式碼示範了如何：
1. **定義一個自訂 Dataset**
2. **初始化並測試 Dataset**
3. **使用 DataLoader 批量載入數據**

這樣的結構可以適用於圖片數據、文本數據等，只需要修改 `__getitem__` 的處理方式即可。