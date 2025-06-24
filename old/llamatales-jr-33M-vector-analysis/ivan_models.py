import sys
sys.path.append("/home/a3murali/clustering/installs")

import os
os.environ['TRANSFORMERS_CACHE'] = '/home/a3murali/hf_cache/'

from huggingface_hub import login
login(token = "<hf_token>")

import transformers
import torch

print(f"PyTorch Version: {torch.__version__}")
print(f"Transformers Version: {transformers.__version__}")

from transformers import AutoConfig, AutoModel, AutoTokenizer, AutoModelForCausalLM, pipeline, LlamaForCausalLM

config = AutoConfig.from_pretrained("/home/a3murali/hf_cache/models--ivnle--llamatales_jr_8b-lay8-hs512-hd8-33M/snapshots/461f50f0024efb46b94dc68cc850d12d75ecb325")
get_model = AutoModel.from_config(config)
print(get_model.layers)
model = "ivnle/llamatales_jr_8b-lay8-hs512-hd8-33M"
tokenizer = AutoTokenizer.from_pretrained(model)


llamatales_pipeline = pipeline("text-generation", model = model, device = "cuda")

prompt = "Generate me a children's story that has a dragon, a prince, and a princess."

sequences = llamatales_pipeline(
    prompt,
    do_sample = True,
    top_k = 10,
    num_return_sequences = 1,
    max_new_tokens = 100)

print(sequences[0]['generated_text'])