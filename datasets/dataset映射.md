# DataSet分類和映射(mapping)
- map()
- 最適合傳出符合訓練資料格式
- DatasetDisc,Dataset都有此方法

## 將只有一個train的劃分為train,validation,test

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

