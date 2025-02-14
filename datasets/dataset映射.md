# DataSet分類和映射(mapping)
- map()
- 最適合傳出符合訓練資料格式(整合tokenizer)
- DatasetDisc,Dataset都有此方法

## 將只有一個train的分類為train,validation,test

```python
from datasets import load_dataset

datasets = load_dataset('soda-lmu/tweet-annotation-sensitivity-2')
datasets

#==output==
DatasetDict({
    train: Dataset({
        features: ['Unnamed: 0', 'case_id', 'duration_seconds', 'last_screen', 'device', 'ethn_hispanic', 'ethn_white', 'ethn_afr_american', 'ethn_asian', 'ethn_sth_else', 'ethn_prefer_not', 'age', 'education', 'english_fl', 'twitter_use', 'socmedia_use', 'prolific_hours', 'task_fun', 'task_interesting', 'task_boring', 'task_repetitive', 'task_important', 'task_depressing', 'task_offensive', 'repeat_tweet_coding', 'repeat_hs_coding', 'target_online_harassment', 'target_other_harassment', 'party_affiliation', 'societal_relevance_hs', 'annotator_id', 'condition', 'tweet_batch', 'hate_speech', 'offensive_language', 'tweet_id', 'orig_label_hs', 'orig_label_ol', 'orig_label_ne', 'tweet_hashed'],
        num_rows: 89150
    })
})
```

**取出全部**

```python
from datasets import load_dataset,DatasetDict

datasets = load_dataset('soda-lmu/tweet-annotation-sensitivity-2')
dataset = datasets['train']
dataset

#==output==
Dataset({
    features: ['Unnamed: 0', 'case_id', 'duration_seconds', 'last_screen', 'device', 'ethn_hispanic', 'ethn_white', 'ethn_afr_american', 'ethn_asian', 'ethn_sth_else', 'ethn_prefer_not', 'age', 'education', 'english_fl', 'twitter_use', 'socmedia_use', 'prolific_hours', 'task_fun', 'task_interesting', 'task_boring', 'task_repetitive', 'task_important', 'task_depressing', 'task_offensive', 'repeat_tweet_coding', 'repeat_hs_coding', 'target_online_harassment', 'target_other_harassment', 'party_affiliation', 'societal_relevance_hs', 'annotator_id', 'condition', 'tweet_batch', 'hate_speech', 'offensive_language', 'tweet_id', 'orig_label_hs', 'orig_label_ol', 'orig_label_ne', 'tweet_hashed'],
    num_rows: 89150
})
```

**分割80%,20%**

```python
another_dict = dataset.train_test_split(test_size=0.2,seed=42)
test_dataset = another_dict['test']
test_dataset

#==output==
Dataset({
    features: ['Unnamed: 0', 'case_id', 'duration_seconds', 'last_screen', 'device', 'ethn_hispanic', 'ethn_white', 'ethn_afr_american', 'ethn_asian', 'ethn_sth_else', 'ethn_prefer_not', 'age', 'education', 'english_fl', 'twitter_use', 'socmedia_use', 'prolific_hours', 'task_fun', 'task_interesting', 'task_boring', 'task_repetitive', 'task_important', 'task_depressing', 'task_offensive', 'repeat_tweet_coding', 'repeat_hs_coding', 'target_online_harassment', 'target_other_harassment', 'party_affiliation', 'societal_relevance_hs', 'annotator_id', 'condition', 'tweet_batch', 'hate_speech', 'offensive_language', 'tweet_id', 'orig_label_hs', 'orig_label_ol', 'orig_label_ne', 'tweet_hashed'],
    num_rows: 17830
})
```

**將20%,再分割為50%,50%**

```python
validation_test_dict = test_dataset.train_test_split(test_size=0.5,seed=42)
validation_test_dict

#==output==
DatasetDict({
    train: Dataset({
        features: ['Unnamed: 0', 'case_id', 'duration_seconds', 'last_screen', 'device', 'ethn_hispanic', 'ethn_white', 'ethn_afr_american', 'ethn_asian', 'ethn_sth_else', 'ethn_prefer_not', 'age', 'education', 'english_fl', 'twitter_use', 'socmedia_use', 'prolific_hours', 'task_fun', 'task_interesting', 'task_boring', 'task_repetitive', 'task_important', 'task_depressing', 'task_offensive', 'repeat_tweet_coding', 'repeat_hs_coding', 'target_online_harassment', 'target_other_harassment', 'party_affiliation', 'societal_relevance_hs', 'annotator_id', 'condition', 'tweet_batch', 'hate_speech', 'offensive_language', 'tweet_id', 'orig_label_hs', 'orig_label_ol', 'orig_label_ne', 'tweet_hashed'],
        num_rows: 8915
    })
    test: Dataset({
        features: ['Unnamed: 0', 'case_id', 'duration_seconds', 'last_screen', 'device', 'ethn_hispanic', 'ethn_white', 'ethn_afr_american', 'ethn_asian', 'ethn_sth_else', 'ethn_prefer_not', 'age', 'education', 'english_fl', 'twitter_use', 'socmedia_use', 'prolific_hours', 'task_fun', 'task_interesting', 'task_boring', 'task_repetitive', 'task_important', 'task_depressing', 'task_offensive', 'repeat_tweet_coding', 'repeat_hs_coding', 'target_online_harassment', 'target_other_harassment', 'party_affiliation', 'societal_relevance_hs', 'annotator_id', 'condition', 'tweet_batch', 'hate_speech', 'offensive_language', 'tweet_id', 'orig_label_hs', 'orig_label_ol', 'orig_label_ne', 'tweet_hashed'],
        num_rows: 8915
    })
})
```

**將全部整合為一個DatasetDict**

```python
datasets_3kind = DatasetDict({
    "train": another_dict['train'],
    "validation": validation_test_dict['train'],
    "test": validation_test_dict['test']
})

datasets_3kind

#==output==
DatasetDict({
    train: Dataset({
        features: ['Unnamed: 0', 'case_id', 'duration_seconds', 'last_screen', 'device', 'ethn_hispanic', 'ethn_white', 'ethn_afr_american', 'ethn_asian', 'ethn_sth_else', 'ethn_prefer_not', 'age', 'education', 'english_fl', 'twitter_use', 'socmedia_use', 'prolific_hours', 'task_fun', 'task_interesting', 'task_boring', 'task_repetitive', 'task_important', 'task_depressing', 'task_offensive', 'repeat_tweet_coding', 'repeat_hs_coding', 'target_online_harassment', 'target_other_harassment', 'party_affiliation', 'societal_relevance_hs', 'annotator_id', 'condition', 'tweet_batch', 'hate_speech', 'offensive_language', 'tweet_id', 'orig_label_hs', 'orig_label_ol', 'orig_label_ne', 'tweet_hashed'],
        num_rows: 71320
    })
    validation: Dataset({
        features: ['Unnamed: 0', 'case_id', 'duration_seconds', 'last_screen', 'device', 'ethn_hispanic', 'ethn_white', 'ethn_afr_american', 'ethn_asian', 'ethn_sth_else', 'ethn_prefer_not', 'age', 'education', 'english_fl', 'twitter_use', 'socmedia_use', 'prolific_hours', 'task_fun', 'task_interesting', 'task_boring', 'task_repetitive', 'task_important', 'task_depressing', 'task_offensive', 'repeat_tweet_coding', 'repeat_hs_coding', 'target_online_harassment', 'target_other_harassment', 'party_affiliation', 'societal_relevance_hs', 'annotator_id', 'condition', 'tweet_batch', 'hate_speech', 'offensive_language', 'tweet_id', 'orig_label_hs', 'orig_label_ol', 'orig_label_ne', 'tweet_hashed'],
        num_rows: 8915
    })
    test: Dataset({
        features: ['Unnamed: 0', 'case_id', 'duration_seconds', 'last_screen', 'device', 'ethn_hispanic', 'ethn_white', 'ethn_afr_american', 'ethn_asian', 'ethn_sth_else', 'ethn_prefer_not', 'age', 'education', 'english_fl', 'twitter_use', 'socmedia_use', 'prolific_hours', 'task_fun', 'task_interesting', 'task_boring', 'task_repetitive', 'task_important', 'task_depressing', 'task_offensive', 'repeat_tweet_coding', 'repeat_hs_coding', 'target_online_harassment', 'target_other_harassment', 'party_affiliation', 'societal_relevance_hs', 'annotator_id', 'condition', 'tweet_batch', 'hate_speech', 'offensive_language', 'tweet_id', 'orig_label_hs', 'orig_label_ol', 'orig_label_ne', 'tweet_hashed'],
        num_rows: 8915
    })
})
```

## 資料集的映射
- map()

**Dataset執行映射**

```python
test_dataset = datasets_3kind['test']
test_dataset['tweet_hashed'][:5]

>==output==
["@###### I'll show up and chug a beer or 8",
 "Reading the ut sexual assault case takes me back 2 college.I'll never forget D Walk would say dont run trains on them hoes.Straight Trouble",
 '@###### Aww y u so mad tho, a successful man LMFAO, hut hugging as faggot.',
 "The people who have historically been called white trash were called that for a reason. They're white trash. The truth hurts.",
 "The family of Ron O'Neal needs to sue Wu-Tang for that trash song they released in his name."]
```

```python
def add_prefix(example):    
    example['tweet_hashed'] = "prefix:" + example['tweet_hashed']
    return example
   
#因為tweet_hashed有None,先清理資料
test_dataset = test_dataset.filter(lambda example: example['tweet_hashed'] is not None)
prefix_test_dataset = test_dataset.map(add_prefix)
prefix_test_dataset['tweet_hashed'][:5]

#==output==
["prefix:@###### I'll show up and chug a beer or 8",
 "prefix:Reading the ut sexual assault case takes me back 2 college.I'll never forget D Walk would say dont run trains on them hoes.Straight Trouble",
 'prefix:@###### Aww y u so mad tho, a successful man LMFAO, hut hugging as faggot.',
 "prefix:The people who have historically been called white trash were called that for a reason. They're white trash. The truth hurts.",
 "prefix:The family of Ron O'Neal needs to sue Wu-Tang for that trash song they released in his name."]
 
```

**DatasetDict使用map**

```poython
print(test_dataset['education'][:5])
print(test_dataset['tweet_hashed'][:5])
```

```python
def add_prefix1(example): 
    if example['education'] is not None and example['tweet_hashed'] is not None:   
        example['tweet_hashed'] = "prefix:" + example['tweet_hashed']
    return example

datasets_3kind.map(add_prefix1)

#==output==
DatasetDict({
    train: Dataset({
        features: ['Unnamed: 0', 'case_id', 'duration_seconds', 'last_screen', 'device', 'ethn_hispanic', 'ethn_white', 'ethn_afr_american', 'ethn_asian', 'ethn_sth_else', 'ethn_prefer_not', 'age', 'education', 'english_fl', 'twitter_use', 'socmedia_use', 'prolific_hours', 'task_fun', 'task_interesting', 'task_boring', 'task_repetitive', 'task_important', 'task_depressing', 'task_offensive', 'repeat_tweet_coding', 'repeat_hs_coding', 'target_online_harassment', 'target_other_harassment', 'party_affiliation', 'societal_relevance_hs', 'annotator_id', 'condition', 'tweet_batch', 'hate_speech', 'offensive_language', 'tweet_id', 'orig_label_hs', 'orig_label_ol', 'orig_label_ne', 'tweet_hashed'],
        num_rows: 71320
    })
    validation: Dataset({
        features: ['Unnamed: 0', 'case_id', 'duration_seconds', 'last_screen', 'device', 'ethn_hispanic', 'ethn_white', 'ethn_afr_american', 'ethn_asian', 'ethn_sth_else', 'ethn_prefer_not', 'age', 'education', 'english_fl', 'twitter_use', 'socmedia_use', 'prolific_hours', 'task_fun', 'task_interesting', 'task_boring', 'task_repetitive', 'task_important', 'task_depressing', 'task_offensive', 'repeat_tweet_coding', 'repeat_hs_coding', 'target_online_harassment', 'target_other_harassment', 'party_affiliation', 'societal_relevance_hs', 'annotator_id', 'condition', 'tweet_batch', 'hate_speech', 'offensive_language', 'tweet_id', 'orig_label_hs', 'orig_label_ol', 'orig_label_ne', 'tweet_hashed'],
        num_rows: 8915
    })
    test: Dataset({
        features: ['Unnamed: 0', 'case_id', 'duration_seconds', 'last_screen', 'device', 'ethn_hispanic', 'ethn_white', 'ethn_afr_american', 'ethn_asian', 'ethn_sth_else', 'ethn_prefer_not', 'age', 'education', 'english_fl', 'twitter_use', 'socmedia_use', 'prolific_hours', 'task_fun', 'task_interesting', 'task_boring', 'task_repetitive', 'task_important', 'task_depressing', 'task_offensive', 'repeat_tweet_coding', 'repeat_hs_coding', 'target_online_harassment', 'target_other_harassment', 'party_affiliation', 'societal_relevance_hs', 'annotator_id', 'condition', 'tweet_batch', 'hate_speech', 'offensive_language', 'tweet_id', 'orig_label_hs', 'orig_label_ol', 'orig_label_ne', 'tweet_hashed'],
        num_rows: 8915
    })
})
```

**整合tokenizer**

```
from datasets import load_dataset,DatasetDict

datasets = load_dataset('erhwenkuo/wikinews-zhtw')
datasets

#==output==
DatasetDict({
    train: Dataset({
        features: ['id', 'url', 'title', 'text'],
        num_rows: 9827
    })
})
```


```
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('google-bert/bert-base-chinese')
def preprocess_func(example):      
    model_inputs = tokenizer(example['text'], max_length=512, truncation=True)
    model_inputs['labels'] = tokenizer(example['title'], max_length=200, truncation=True) 
    return model_inputs
    

datasets.map(preprocess_func,remove_columns=datasets['train'].column_names)
```

