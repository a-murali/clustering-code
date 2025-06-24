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
model = AutoModelForCausalLM.from_config(config).to('cuda')
print(model)
#print(type(model))
#print(model.layers)

for i in model.named_parameters():
    print(f"{i[0]} -> {i[1].device}")

tokenizer = AutoTokenizer.from_pretrained("/home/a3murali/hf_cache/models--ivnle--llamatales_jr_8b-lay8-hs512-hd8-33M/snapshots/461f50f0024efb46b94dc68cc850d12d75ecb325/", config = config, padding_side = 'left')
tokenizer.pad_token = tokenizer.eos_token = '<pad>'


prompt = "Generate me a children's story that has a dragon, a prince, and a princess."
inputs = tokenizer(prompt, padding = True, return_tensors="pt").to("cuda")

outputs = model.generate(inputs.input_ids, attention_mask=inputs["attention_mask"].to('cuda'), do_sample = False, top_k = 10, num_return_sequences = 1, max_new_tokens = 100)
#outputs = model.generate(inputs.input_ids, attention_mask=inputs["attention_mask"], do_sample = False, top_k = 10, num_return_sequences = 1, max_new_tokens = 100, eos_token_id = tokenizer.eos_token_id)#, pad_token_id = tokenizer.pad_token_id)

print(tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0])
# messages = [
#     {
#         "role": "system",
#         "content": "You are a friendly chatbot who always responds in the style of a pirate",
#     },
#     {"role": "user", "content": prompt},
# ]

# tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to('cuda')

# print(tokenizer.decode(tokenized_chat[0]))

# outputs = model.generate(tokenized_chat, max_new_tokens=128)
# print(tokenizer.decode(outputs[0]))