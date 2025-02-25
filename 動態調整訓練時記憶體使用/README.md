# 動態調整訓練時記憶體使用
**載入數據集資料**

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
'''
dataset_dict = load_dataset("csv",data_files="./ChnSentiCorp_htl_all.csv") #split不指定會傳出DatasetDict
dataset_dict['train']
'''
#整合上面2行成為1行的語法
dataset = load_dataset("csv",data_files='./ChnSentiCorp_htl_all.csv', split="train")
dataset
#清理資料
dataset = dataset.filter(lambda example: example['review'] is not None)
dataset

#==output==
Dataset({
    features: ['label', 'review'],
    num_rows: 7765
})
```

**數據集分類**

```python
datasets = dataset.train_test_split(test_size=0.1)
datasets

#==output==
DatasetDict({
    train: Dataset({
        features: ['label', 'review'],
        num_rows: 6988
    })
    test: Dataset({
        features: ['label', 'review'],
        num_rows: 777
    })
})
```

**數據集分詞處理**

```python
tokenizer = AutoTokenizer.from_pretrained('google-bert/bert-base-chinese')

def tokenizer_process(example:dict[str,any]) -> dict[str,any]:
    tokenized_example:dict = tokenizer(example['review'], max_length=128, truncation=True)
    tokenized_example['labels'] = example['label']
    return tokenized_example

tokenized_datasets = datasets.map(tokenizer_process,batched=True,remove_columns=datasets['train'].column_names)
tokenized_datasets

#==output==
DatasetDict({
    train: Dataset({
        features: ['input_ids', 'token_type_ids', 'attention_mask', 'labels'],
        num_rows: 6988
    })
    test: Dataset({
        features: ['input_ids', 'token_type_ids', 'attention_mask', 'labels'],
        num_rows: 777
    })
})
```

**取得預訓練模型**

```python
model = AutoModelForSequenceClassification.from_pretrained('google-bert/bert-base-chinese')
```

**建立評估函數**

```python
import evaluate
#acc_metric = evaluate.load('accuracy')
#f1_metric = evaluate.load('f1')
acc_metric = evaluate.load('evaluate-main/metrics/accuracy/accuracy.py')
f1_metric = evaluate.load('evaluate-main/metrics/f1/f1.py')

def eval_metric(eval_predict):
    predictions, labels = eval_predict
    predictions = predictions.argmax(axis=-1)
    acc = acc_metric.compute(predictions=predictions, references=labels)
    f1 = f1_metric.compute(predictions=predictions, references=labels)
    acc.update(f1)
    return acc

```

**建立TrainingArguments**

```python
train_args = TrainingArguments(
    output_dir='./checkpoints',
    per_device_train_batch_size=64,
    per_device_eval_batch_size=128,
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=3,
    learning_rate=2e-5,
    weight_decay=0.01,
    metric_for_best_model='f1',
    load_best_model_at_end=True,
    report_to='None',
    num_train_epochs=3 #預設為3
)
```

**TrainingArguments說明**

TrainingArguments 是 Hugging Face transformers 中 Trainer 類別的設定參數，主要用於控制模型訓練的各種行為。讓我們詳細解析你的 TrainingArguments 設定：

1. 輸出目錄 (output_dir)

output_dir='./checkpoints'

	•	作用：指定訓練過程中保存模型檔案的位置。
	•	原因：這樣可以在訓練過程中自動保存檢查點（checkpoints），以便在訓練中斷時能夠繼續訓練，或用來做推論（inference）。
	•	注意：最好確保這個目錄存在，並且有足夠的磁碟空間來存放權重檔案。

2. 每個裝置的訓練批次大小 (per_device_train_batch_size)

per_device_train_batch_size=64

	•	作用：設定單個 GPU（或 CPU）上的訓練批次大小。
	•	原因：較大的批次大小有助於更穩定的梯度更新，但會消耗更多的記憶體（VRAM）。64 是一個較大的值，適合高效能 GPU。
	•	調整建議：
	•	若 GPU 記憶體不足，應降低此值，如 32 或 16。
	•	若使用多個 GPU，總批次大小 = per_device_train_batch_size × GPU 數量。

3. 每個裝置的評估批次大小 (per_device_eval_batch_size)

per_device_eval_batch_size=128

	•	作用：設定單個 GPU（或 CPU）上的評估批次大小。
	•	原因：在評估時不需要計算梯度，因此可以使用更大的批次大小來加快評估速度。
	•	調整建議：
	•	如果顯示記憶體充足，可以適當提高這個值來加速評估。

4. 記錄 (logging_steps)

logging_steps=10

	•	作用：設定每 10 個步驟記錄一次訓練指標（如 loss）。
	•	原因：讓使用者可以監控訓練進度，而不會因為太頻繁的記錄而影響效能。
	•	調整建議：
	•	若想要更細緻的監控，可減少這個值（如 5）。
	•	若訓練步驟過多，可能需要增加此值來減少 log 頻率。

5. 評估策略 (evaluation_strategy)

evaluation_strategy="epoch"

	•	作用：設定評估（validation）的頻率。
	•	可選值：
	•	"no"：不做評估。
	•	"steps"：每隔 eval_steps 設定的步驟進行一次評估。
	•	"epoch"：每個 epoch 結束時進行評估。
	•	原因：這裡選擇 "epoch"，表示在每個完整的訓練週期結束後，執行一次評估，以確保模型的訓練效果。

6. 模型儲存策略 (save_strategy)

save_strategy="epoch"

	•	作用：設定模型的儲存頻率。
	•	可選值：
	•	"no"：不儲存模型。
	•	"steps"：每 save_steps 設定的步驟存一次。
	•	"epoch"：每個完整的 epoch 後存一次。
	•	原因：與 evaluation_strategy 一致，每個 epoch 後存一次最佳檢查點，方便後續微調或恢復訓練。

7. 最大保存的檢查點數量 (save_total_limit)

save_total_limit=3

	•	作用：最多保留 3 個檢查點，超過這個數量後會自動刪除舊的檢查點。
	•	原因：
	•	若不限制，可能會占用大量磁碟空間。
	•	3 代表保留最近 3 次的最佳模型，足夠進行回溯與選擇。

8. 學習率 (learning_rate)

learning_rate=2e-5

	•	作用：設定 AdamW 優化器的學習率（Learning Rate, LR）。
	•	原因：
	•	2e-5（0.00002）是適合 Transformer 模型的預設微調學習率。
	•	若學習率過高，模型可能難以收斂（loss 波動大）。
	•	若學習率過低，訓練速度變慢。
	•	調整建議：
	•	若模型訓練不穩定，可嘗試降低，如 1e-5 或 5e-6。
	•	若模型訓練過慢且 loss 平穩，可提高學習率，如 3e-5。

9. 權重衰減 (weight_decay)

weight_decay=0.01

	•	作用：L2 正則化，用於防止過擬合，讓權重更新時適當衰減。
	•	原因：
	•	0.01 是較常見的 Transformer 權重衰減值。
	•	若模型容易過擬合，可以適當提高，如 0.02。
	•	若模型學習緩慢或效果不好，可以降低此值，如 0.001。

10. 最佳模型的評估指標 (metric_for_best_model)

metric_for_best_model='f1'

	•	作用：設定用來選擇最佳模型的評估指標。
	•	原因：
	•	f1 適用於不平衡數據集，因為它是 Precision 和 Recall 的加權平均。
	•	若是回歸問題，可改為 mse 或 mae。
	•	若是分類問題，也可使用 accuracy。

11. 載入最佳模型 (load_best_model_at_end)

load_best_model_at_end=True

	•	作用：訓練結束時，自動加載評估指標最高（最佳）的模型。
	•	原因：
	•	可確保最後得到的是最佳檢查點，而不是最後一個檢查點（因為最後一個可能不是最好的）。
	•	如果 metric_for_best_model 是 f1，那麼這個選項會加載 f1 最高的模型。

總結

參數	作用	設定值	說明
output_dir	儲存檢查點的目錄	'./checkpoints'	避免訓練中斷後丟失模型\

per_device_train_batch_size	訓練批次大小	64	取決於 GPU 記憶體大小

per_device_eval_batch_size	評估批次大小	128	評估時可設較大值

logging_steps	訓練日誌頻率	10	控制 log 的頻率

evaluation_strategy	訓練期間何時評估	"epoch"	每個 epoch 結束後評估

save_strategy	訓練期間何時存模型	"epoch"	每個 epoch 後存

save_total_limit	保留多少個檢查點	3	避免磁碟空間不足

learning_rate	學習率	2e-5	Transformer 微調的常見值

weight_decay	權重衰減	0.01	防止過擬合

metric_for_best_model	最佳模型的指標	'f1'	適用於分類任務

load_best_model_at_end	訓練結束後是否載入最佳模型	True	取最好的模型

這些參數設定適合 Transformer 微調，但可以根據硬體資源與數據集特性進行調整。
---

**登入hugging face**

```python
from huggingface_hub import login
login()
```

**上傳model和tokenizer至huggingface**

```python
trainer.push_to_hub("roberthsu2003") #由於有設./checkpoints,所以自動產生checkpoints的repo,也會自動上傳評估至repo
#同時要上傳tokenizer
model_name = "roberthsu2003/checkpoints"
tokenizer.push_to_hub(model_name)
```

**下載和使用**

```python
from transformers import pipeline


id2_label = {'LABEL_0':"負評",'LABEL_1':"正評"}
pipe = pipeline('text-classification', model="roberthsu2003/checkpoints")

sen="服務人員都很親切"
print(sen,id2_label[pipe(sen)[0]['label']])

sen1="服務人員都不親切"
print(sen1,id2_label[pipe(sen1)[0]['label']])

#==output==
服務人員都很親切 正評
服務人員都不親切 負評
```