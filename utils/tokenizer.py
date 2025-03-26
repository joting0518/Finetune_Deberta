import torch
import transformers
from transformers import DebertaV2TokenizerFast
print(f"Torch version: {torch.__version__}")
print("Transformers module is successfully imported!")

def load_tokenizer(model_name: str):
    return DebertaV2TokenizerFast.from_pretrained(model_name)
