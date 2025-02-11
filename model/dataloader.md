**torch.utils.data.DataLoader 的功能與簡單範例**

torch.utils.data.DataLoader 是 PyTorch 中用來加載數據的工具，主要用於批量處理數據、隨機打亂數據、以及並行加載數據（利用多線程加速）。通常與 torch.utils.data.Dataset 搭配使用，以便有效地組織數據。

**功能**


1. **批量處理 (Batching)**

• 透過 batch_size 設定每次讀取的數據量，使模型能夠更高效地處理數據。


2. **隨機打亂 (Shuffling)**

• 使用 shuffle=True 來打亂數據，提升模型的泛化能力。


3. **多線程並行加載 (Parallel Loading)**

• 設置 num_workers 來啟用多線程加載數據，加快數據讀取速度。


4. **數據增強 (Transformations)**

• 可搭配 torchvision.transforms（如影像處理）來預處理數據。


5. **自定義數據集 (Custom Dataset)**

• 可以繼承 torch.utils.data.Dataset 來定義自己的數據讀取方式。

**簡單範例**

1️⃣ 讀取 Tensor 數據

```other
import torch
from torch.utils.data import DataLoader, TensorDataset

# 創建簡單的 Tensor 數據集
x = torch.arange(10).float().unsqueeze(1)  # 生成 10 個數據點
y = x * 2                                  # 標籤為 x 的 2 倍
dataset = TensorDataset(x, y)              # 創建數據集

# 使用 DataLoader 讀取數據
dataloader = DataLoader(dataset, batch_size=3, shuffle=True)

# 迭代數據
for batch in dataloader:
    x_batch, y_batch = batch
    print(f"x_batch: {x_batch.squeeze().tolist()}, y_batch: {y_batch.squeeze().tolist()}")
```


🔹 重點

• 使用 TensorDataset 來封裝 x, y 數據。

• DataLoader 設定 batch_size=3，讓每次讀取 3 筆數據。

• shuffle=True 讓數據順序隨機化。

2️⃣ 讀取自定義數據集

如果數據是從 CSV、圖片或其他來源獲取，通常需要自定義 Dataset。

```other
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self):
        self.data = list(range(1, 11))  # 假設有 10 筆數據
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        y = x * 2
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# 創建自定義數據集
dataset = CustomDataset()
dataloader = DataLoader(dataset, batch_size=3, shuffle=True)

# 迭代數據
for x_batch, y_batch in dataloader:
    print(f"x_batch: {x_batch.tolist()}, y_batch: {y_batch.tolist()}")
```


🔹 重點

• **len** 定義數據集的大小。

• **getitem** 返回一個數據樣本 (x, y)。

• DataLoader 仍然負責批量加載數據。

3️⃣ 讀取圖片數據（使用 torchvision）

對於圖片數據，可搭配 torchvision.datasets 和 transforms 使用：

```other
from torchvision import datasets, transforms

# 定義數據轉換（標準化、轉為 Tensor）
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# 下載並加載 MNIST 數據集
dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# 讀取一個 batch 的數據
images, labels = next(iter(dataloader))
print(f"批量圖片張數: {images.shape}, 批量標籤: {labels.shape}")
```


🔹 重點

• transforms 進行圖片預處理（轉 Tensor、標準化）。

• 直接使用 torchvision.datasets.MNIST 來下載數據。

• batch_size=64，每次讀取 64 張圖片。

**總結**

• DataLoader 是 PyTorch 提供的高效數據加載工具，支援 **批量讀取、隨機打亂、多線程加載**。

• 可以配合 TensorDataset 或 Dataset 來讀取不同類型的數據（如 Tensor、圖片、CSV）。

• 可以使用 torchvision 處理影像數據，搭配 transforms 進行預處理。

這些功能讓訓練神經網絡更加方便和高效！ 🚀
