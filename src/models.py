from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from transformers import pipeline
import streamlit as st

#bart
@st.cache_resource
def load_bart():
    bart_model_name="facebook/bart-large-mnli"
    return pipeline("zero-shot-classification",model=bart_model_name)

#qwen 1.5b
@st.cache_resource
def load_qwen1():
    model_name="Qwen/Qwen2.5-1.5B-Instruct"
    tokenizer=AutoTokenizer.from_pretrained(model_name)
    model=AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()

    return tokenizer,model

#qwen 3
@st.cache_resource
def load_qwen3():
    model_name="Qwen/Qwen2.5-3B-Instruct"
    tokenizer=AutoTokenizer.from_pretrained(model_name)
    model=AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()

    return tokenizer,model

classifier=load_bart()
qwen1_tokenizer,qwen1_model=load_qwen1()
qwen3_tokenizer,qwen3_model=load_qwen3()