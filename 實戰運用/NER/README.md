# 命名實體識別(Named Entity Recognition)
- token-classfication
- [實作ipynb](./ner1.ipynb)


### 載入相關套件
- 必需安裝一個特別的評估套件seqeval
- [seqeval補充說明](./補充說明/seqeval說明.md)

```bash
pip intall seqeval
```

```python
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification
import evaluate
from datasets import load_dataset
```

### 使用數據集

```python

```