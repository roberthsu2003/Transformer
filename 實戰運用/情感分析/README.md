# Trainer和TrainingArguments訓練器
- Trainer class 提供支援PyTorch完整的訓練函數
- Trainer必需配合TrainingArguments
- 會產生評估邏輯

> [!TIP]
> [train實作](./trainer.ipynb)  
> [訓練完成的模型](https://huggingface.co/roberthsu2003/for_classification)  
> 資料集來源: roberthsu2003/data_for_classification  
> 預訓練模型: google-bert/bert-base-chinese

## Trainer的使用限制

- 模型必需回傳tuples或ModelOutput的子類別

- 提供了labels參數，則模型可以計算loss，並且該loss會作為tuple的第一個元素返回（如果您的模型傳回tuple）
- 您的模型可以接受多個標籤參數（在TrainingArguments 中使用 label_names 來向 Trainer 指示它們的名稱），但它們都不應以 「label」 命名。

## [train實作](./trainer.ipynb)





