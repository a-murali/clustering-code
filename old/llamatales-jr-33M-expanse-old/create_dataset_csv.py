import sys
sys.path.append("/home/a3murali/clustering/installs")

import os
os.environ['TRANSFORMERS_CACHE'] = '/home/a3murali/hf_cache/'

from huggingface_hub import login
login(token = "<hf_token>")

import transformers
import torch

import csv
import numpy as np
import pandas as pd

print(f"PyTorch Version: {torch.__version__}")
print(f"Transformers Version: {transformers.__version__}")

from transformers import AutoConfig, AutoModel, AutoTokenizer, AutoModelForCausalLM, pipeline, LlamaForCausalLM

config = AutoConfig.from_pretrained("/home/a3murali/hf_cache/models--ivnle--llamatales_jr_8b-lay8-hs512-hd8-33M/snapshots/461f50f0024efb46b94dc68cc850d12d75ecb325",  output_hidden_states = True)
get_model = AutoModel.from_config(config)
print(get_model.layers)
model = "ivnle/llamatales_jr_8b-lay8-hs512-hd8-33M"
tokenizer = AutoTokenizer.from_pretrained(model)


llamatales_pipeline = pipeline("text-generation", model = model, device = "cuda")

# prompts = {
#     1: "Once upon a time there was a dragon", 
#     2: "Once upon a time there was a princess", 
#     3: "Once upon a time there were two children",
#     4: "Once upon a time there was a prince",
#     5: "Once upon a time there was a frog",
#     6: "Once upon a time there was a king",
#     7: "Once upon a time there was a queen",
#     8: "Once upon a time there was a wolf",
#     9: "Once upon a time there was a genie",
#     10: "Once upon a time there was a poor boy"
# }
prompts = {1: "Once upon a time there was a dragon"}

data = []
for prompt_id in prompts:
    #generation - generate stories for each prompt
    for i in range(2):
        sequences = llamatales_pipeline(
            prompts[prompt_id],
            do_sample = True,
            top_k = 10,
            num_return_sequences = 1,
            max_new_tokens = 512
        )
        generated_story = sequences[0]['generated_text']
        print(generated_story)

        num_tokens_generated_story = len(tokenizer.encode(generated_story))

        #test - collect hidden states
        
        config = AutoConfig.from_pretrained("/home/a3murali/hf_cache/models--ivnle--llamatales_jr_8b-lay8-hs512-hd8-33M/snapshots/461f50f0024efb46b94dc68cc850d12d75ecb325", output_hidden_states = True)
        model = AutoModelForCausalLM.from_config(config).to('cuda')
        print(model)

        for i in model.named_parameters():
            print(f"{i[0]} -> {i[1].device}")


        tokenizer = AutoTokenizer.from_pretrained("/home/a3murali/hf_cache/models--ivnle--llamatales_jr_8b-lay8-hs512-hd8-33M/snapshots/461f50f0024efb46b94dc68cc850d12d75ecb325/", config = config)

        inputs = tokenizer(generated_story, return_tensors="pt").to("cuda")

        outputs = model.generate(inputs.input_ids, attention_mask=inputs["attention_mask"], do_sample = True, top_k = 10, num_return_sequences = 1, max_new_tokens = 512, eos_token_id = tokenizer.eos_token_id, pad_token_id = tokenizer.pad_token_id)

        #analysis - collect prompt id of each token
        output_id = []
        for token in range(num_tokens_generated_story):
            output_id.append(prompt_id)
        
        convert_hidden_states = []

        for i in outputs.hidden_states:
            convert_hidden_states.append([j.detach().cpu().numpy() for j in i])
        
        print(len(convert_hidden_states))
        print(len(convert_hidden_states[0]))
        print(len(convert_hidden_states[0][0]))
        print(len(convert_hidden_states[0][0][0]))
        print(len(convert_hidden_states[0][0][0][0]))
        #print(len(convert_hidden_states[0][0][0][0][0]))
        data.append([prompts[prompt_id], generated_story, convert_hidden_states, np.array(output_id)])

        #(([]))

df = pd.DataFrame(data, columns = ["prompt", "story", "hidden_states", "output_token_prompt_id"], dtype = 'object')
print(df)
df.to_csv("story_dataset.csv", index = False)
