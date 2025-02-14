# DataSet一般操作
## 下載DataSets

**單1資料集**

```python
from datasets import load_dataset
datasets = load_dataset('AWeirdDev/zh-tw-recipes-sm')
datasets

#==output==
DatasetDict({
    train: Dataset({
        features: ['image', 'title', 'descriotion', 'cooking_time', 'author', 'url', 'servings', 'ingredients', 'steps'],
        num_rows: 1799
    })
})
```

**多重資料集**

```python
from datasets import load_dataset
dataset = load_dataset('nyu-mll/glue',data_dir='ax')
dataset

#==output==
DatasetDict({
    test: Dataset({
        features: ['premise', 'hypothesis', 'label', 'idx'],
        num_rows: 1104
    })
})
```

**依照資料劃分下載**

```python
from datasets import load_dataset
dataset = load_dataset('nyu-mll/glue',data_dir="mrpc",split='validation')
dataset

#==output==
Dataset({
    features: ['sentence1', 'sentence2', 'label', 'idx'],
    num_rows: 408
})
```

```python
from datasets import load_dataset
dataset = load_dataset('nyu-mll/glue',data_dir="mrpc",split='train[:100]')
dataset

#==output==
from datasets import load_dataset
dataset = load_dataset('nyu-mll/glue',data_dir="mrpc",split='train[:100]')
dataset
```

```python
from datasets import load_dataset
dataset = load_dataset('nyu-mll/glue',data_dir="mrpc",split='train[:50%]')
dataset

#==output==
Dataset({
    features: ['sentence1', 'sentence2', 'label', 'idx'],
    num_rows: 1834
})
```

## 查看DataSets

```python
from datasets import load_dataset
datasets = load_dataset('AWeirdDev/zh-tw-recipes-sm')
datasets

#==output==
DatasetDict({
    train: Dataset({
        features: ['image', 'title', 'descriotion', 'cooking_time', 'author', 'url', 'servings', 'ingredients', 'steps'],
        num_rows: 1799
    })
})
```

```python
datasets['train'][0]

#==output==
{'image': 'https://imageproxy.icook.network/resize?background=255%2C255%2C255&height=150&nocrop=false&stripmeta=true&type=auto&url=http%3A%2F%2Ftokyo-kitchen.icook.tw.s3.amazonaws.com%2Fuploads%2Frecipe%2Fcover%2F455340%2F7e9435a22d6413f3.jpg&width=200',
 'title': '可樂洋蔥炒牛五花',
 'descriotion': '簡單、快速、美味',
 'cooking_time': None,
 'author': '簡單煮',
 'url': 'https://icook.tw/recipes/455340',
 'servings': None,
 'ingredients': [{'name': '冷凍牛五花火鍋肉片', 'unit': '12片'},
  {'name': '洋蔥切片', 'unit': '1顆'},
  {'name': '蔥花（可不加）', 'unit': '適量'},
  {'name': '油', 'unit': '10毫升'},
  {'name': '白胡椒鹽', 'unit': '適量'},
  {'name': '可樂', 'unit': '20'},
  {'name': '醬油（7~10ml）', 'unit': '10毫升'},
  {'name': '酒', 'unit': '10毫升'}],
 'steps': '1. 備料：\n1.肉片解凍備用\n2.洋蔥切片、切蔥花\n料理步驟：\n使用炒菜鍋，熱鍋加油，下肉片炒至微白，起鍋備用➡️加洋蔥，炒2分鐘➡️加炒過的肉片+調味醬汁+適量白胡椒鹽+蔥花（可不加），炒至醬汁略為收乾，就完成了'}
```

```python
datasets['train'][:2]

#==output==
{'image': ['https://imageproxy.icook.network/resize?background=255%2C255%2C255&height=150&nocrop=false&stripmeta=true&type=auto&url=http%3A%2F%2Ftokyo-kitchen.icook.tw.s3.amazonaws.com%2Fuploads%2Frecipe%2Fcover%2F455340%2F7e9435a22d6413f3.jpg&width=200',
  'https://imageproxy.icook.network/resize?background=255%2C255%2C255&height=150&nocrop=false&stripmeta=true&type=auto&url=http%3A%2F%2Ftokyo-kitchen.icook.tw.s3.amazonaws.com%2Fuploads%2Frecipe%2Fcover%2F455332%2Fdf1da3e7af336fe3.jpg&width=200'],
 'title': ['可樂洋蔥炒牛五花', '古早味海鮮湯麵'],
 'descriotion': ['簡單、快速、美味',
  '回鄉下老家時，每當親朋好友來拜訪，就會煮一大鍋的”古早味海鮮湯麵”，當正餐、點心或消夜招待，今天以這道古早味海鮮湯麵，加上『「烹大...'],
 'cooking_time': [None, None],
 'author': ['簡單煮', '廖芬芳'],
 'url': ['https://icook.tw/recipes/455340', 'https://icook.tw/recipes/455332'],
 'servings': [None, None],
 'ingredients': [[{'name': '冷凍牛五花火鍋肉片', 'unit': '12片'},
   {'name': '洋蔥切片', 'unit': '1顆'},
   {'name': '蔥花（可不加）', 'unit': '適量'},
   {'name': '油', 'unit': '10毫升'},
   {'name': '白胡椒鹽', 'unit': '適量'},
   {'name': '可樂', 'unit': '20'},
   {'name': '醬油（7~10ml）', 'unit': '10毫升'},
   {'name': '酒', 'unit': '10毫升'}],
  [{'name': '鮮蝦', 'unit': '20隻'},
   {'name': '透抽', 'unit': '1尾'},
   {'name': '香菇', 'unit': '8朵'},
   {'name': '蝦米', 'unit': '1小把'},
   {'name': '魚餃', 'unit': '8顆'},
   {'name': '芹菜', 'unit': '2支'},
   {'name': '韮菜', 'unit': '5支'},
   {'name': '高麗菜', 'unit': '1/4顆'},
   {'name': '乾麵條', 'unit': '300克'},
   {'name': '雞蛋', 'unit': '5顆'},
   {'name': '烹大師鰹魚風味', 'unit': '1大匙'},
   {'name': '烏醋', 'unit': '1大匙'},
   {'name': '海鹽', 'unit': '2茶匙'}]],
 'steps': ['1. 備料：\n1.肉片解凍備用\n2.洋蔥切片、切蔥花\n料理步驟：\n使用炒菜鍋，熱鍋加油，下肉片炒至微白，起鍋備用➡️加洋蔥，炒2分鐘➡️加炒過的肉片+調味醬汁+適量白胡椒鹽+蔥花（可不加），炒至醬汁略為收乾，就完成了',
  '1. 備好所有食材。\n2. 起鍋，熱油。\n3. 炸蛋酥。\n4. 炸蛋酥，同時，又起一鍋水煮滾。\n5. 蛋酥炸好撈起，原鍋再爆香，香菇。\n6. 再加入，蝦米爆香。\n7. 再加入，魚餃乾煎。\n8. 香氣撲鼻，再倒入1大匙烏醋嗆味。\n9. 倒入煮沸騰的開水。\n10. 乾麵條下鍋。\n11. 加入，烹大師鰹魚風味1大匙。\n12. 麵鍋滾後，加入高麗菜，拌勻。\n13. 加入，鮮蝦。\n14. 再加入，透抽。\n15. 再加入，蛋酥。\n16. 再加入，韮菜段。\n17. 再加入，芹菜珠，2茶匙海鹽，調味！\n18. 喇喇勒！色香俱全，鮮甜美味⋯⋯\n19. 幸福上桌⋯']}
```

```python
datasets['train']['title'][:5]

#==output==
['可樂洋蔥炒牛五花', '古早味海鮮湯麵', '鹹蛋蝦仁餛飩', '雪Q餅', '雪q餅']
```

**column_names**
```python
datasets['train'].column_names

#==output==
['image',
 'title',
 'descriotion',
 'cooking_time',
 'author',
 'url',
 'servings',
 'ingredients',
 'steps']
```

**features**

```python
datasets['train'].features

#==output==
{'image': Value(dtype='string', id=None),
 'title': Value(dtype='string', id=None),
 'descriotion': Value(dtype='string', id=None),
 'cooking_time': Value(dtype='string', id=None),
 'author': Value(dtype='string', id=None),
 'url': Value(dtype='string', id=None),
 'servings': Value(dtype='int64', id=None),
 'ingredients': [{'name': Value(dtype='string', id=None),
   'unit': Value(dtype='string', id=None)}],
 'steps': Value(dtype='string', id=None)}
```



## 分割DataSets

```python
dataset = datasets['train']
dataset.train_test_split(test_size=0.1)

#==output==
DatasetDict({
    train: Dataset({
        features: ['image', 'title', 'descriotion', 'cooking_time', 'author', 'url', 'servings', 'ingredients', 'steps'],
        num_rows: 1619
    })
    test: Dataset({
        features: ['image', 'title', 'descriotion', 'cooking_time', 'author', 'url', 'servings', 'ingredients', 'steps'],
        num_rows: 180
    })
})
```

## DataSets選取和過濾

- 選取為選取的內容成為一個新的DataSet

**選取0,1成為新的DataSet**

```python
dataset.select([0,1])

#==output==
Dataset({
    features: ['image', 'title', 'descriotion', 'cooking_time', 'author', 'url', 'servings', 'ingredients', 'steps'],
    num_rows: 2
})
```


**過濾(傳回全新的Dataset)**

```python
dataset.filter(lambda ownset: "古早味" in ownset['title'])

#==output==
Dataset({
    features: ['image', 'title', 'descriotion', 'cooking_time', 'author', 'url', 'servings', 'ingredients', 'steps'],
    num_rows: 12
})
```

