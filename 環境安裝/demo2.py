import gradio as gr
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
model = AutoModelForQuestionAnswering.from_pretrained('uer/roberta-base-chinese-extractive-qa')
tokenizer = AutoTokenizer.from_pretrained('uer/roberta-base-chinese-extractive-qa')
qa = pipeline('question-answering',model=model,tokenizer=tokenizer)
demo = gr.Interface.from_pipeline(qa)
demo.launch()