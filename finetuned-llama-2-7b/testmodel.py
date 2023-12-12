from pickle import FALSE
import transformers
import textwrap
from transformers import LlamaTokenizer, LlamaForCausalLM, pipeline
import os
import sys
from typing import List
 
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    PeftModel,
    PeftConfig
)
 
import fire
import torch
from datasets import load_dataset
import pandas as pd
 
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from pylab import rcParams

import json

BASE_MODEL = "NousResearch/Llama-2-7b-hf"
LORA_WEIGHTS = "./finetuned"

# Load the pre-trained model and tokenizer
model = LlamaForCausalLM.from_pretrained(
    BASE_MODEL,
    load_in_8bit=True,
    torch_dtype=torch.float16,
    device_map={"": 0}
)
tokenizer = LlamaTokenizer.from_pretrained(BASE_MODEL)

model = PeftModel.from_pretrained(model, LORA_WEIGHTS)
model.config.use_cache = False

prompt = "Detect the toxicity of the comment.: Bananas are fucking gross."
# Provide a list of candidate labels for text classification
candidate_labels = ["Toxic", "Nontoxic"]  # Replace with your actual labels

# classifier = pipeline(task="text-classification", model=model, tokenizer=tokenizer)
# result = classifier("Detect the toxicity of the comment.", candidate_labels=["Toxic, Nontoxic"])

# print(result)


# pipe = pipeline(task="text-generation", do_sample=True, num_return_sequences=1, eos_token_id=$


pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=200)
result = pipe(f"[INST] {prompt} [/INST]")

print(result[0]['generated_text'])
