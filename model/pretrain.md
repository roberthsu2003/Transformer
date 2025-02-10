# Model的預訓練
- 僅使用tokenizer和model
- 使用原生pytorch做預訓練
- 複雜度較高
- 當使用huggingface的dataset,evaluate,trainer,可以簡單化目前的流程

## 使用的資料集
- [SophonPlus](https://github.com/SophonPlus/ChineseNlpCorpus)
- 資料夾內有提供轉換為繁體中文的檔案(ChnSentiCorp_htl_all.csv)