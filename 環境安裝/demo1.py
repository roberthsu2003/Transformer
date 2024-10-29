import gradio as gr
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
model = AutoModelForSequenceClassification.from_pretrained('uer/roberta-base-finetuned-dianping-chinese')
tokenizer = AutoTokenizer.from_pretrained('uer/roberta-base-finetuned-dianping-chinese')
#classification = pipeline('text-classification',model=model,tokenizer=tokenizer)
classification = pipeline('text-classification',model=model,tokenizer=tokenizer,device=0) #使用mac gpu 或 cuda
gr.Interface.from_pipeline(classification).launch()
