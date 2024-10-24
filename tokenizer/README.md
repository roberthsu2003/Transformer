## Tokenizer

## Tokenizer是什麼?
Transformer 的 Tokenizer 是將文字轉換成適合 Transformer 模型格式的重要元件。它執行標記化，包括將文字分割成較小的單位 (標記)，並將這些標記對應成唯一的整數 ID。

**常見類型的分詞器包括**：

	• 位元組對編碼(BPE)
	• 文字片段
	• 句子片段
	
這些方法可以處理子詞，允許模型透過將看不見的詞分解為已知的組件來處理它們。這種混合方法平衡了詞彙量大小和表示質量，從而提高了模型性能

### 數據預處理
- Step1 分詞:使用分詞器對內容數據進行分詞(字,字詞)
- Step2 建構詞典:根據數據集分詞的結果，建構詞典映射(optioanl, 如果採用預訓練詞向量,詞典暈射要根據詞向量文件進行處理)
- Step3 數據轉換:根據建構好的詞典,將分詞處理後的數據做映射,將本文序列轉換為數字序列
- Step4: 數據填充與截斷:再以batch(批次)輸入到模型的方式中，需要對過短的數據進行填充,過長的數據進行截斷,保證數據長度符合模型能接受的範圍,同時batch內的數據維度大小一致。

### 基本使用
- 載入和保存(from_pretrained / save_pretrained)
- 句子分詞(tokenize)
- 查看詞典(vocab)
- 索引轉換(convert_tokens_to_ids / convert_ids_to_tokens)
- 填充截斷(padding / truncation)
- 其它輸入(attention_mast / token_type_ids)