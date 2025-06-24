import sys
sys.path.append("/home/a3murali/clustering/installs")

import os
os.environ['TRANSFORMERS_CACHE'] = '/home/a3murali/hf_cache/'

from huggingface_hub import login
login(token = "<hf_token>")

import transformers
import torch

import json

print(f"PyTorch Version: {torch.__version__}")
print(f"Transformers Version: {transformers.__version__}")

from transformers import AutoConfig, AutoModel, AutoTokenizer, AutoModelForCausalLM, pipeline, LlamaForCausalLM

config = AutoConfig.from_pretrained("/home/a3murali/hf_cache/models--ivnle--llamatales_jr_8b-lay8-hs512-hd8-33M/snapshots/461f50f0024efb46b94dc68cc850d12d75ecb325",  output_hidden_states = True)
get_model = AutoModel.from_config(config)
print(get_model.layers)
model = "ivnle/llamatales_jr_8b-lay8-hs512-hd8-33M"
tokenizer = AutoTokenizer.from_pretrained(model)


llamatales_pipeline = pipeline("text-generation", model = model, device = "cuda")

prompts = {
    1: "Once upon a time there was a dragon", 
    2: "Once upon a time there was a princess", 
    3: "Once upon a time there were two children",
    4: "Once upon a time there was a prince",
    5: "Once upon a time there was a frog",
    6: "Once upon a time there was a king",
    7: "Once upon a time there was a queen",
    8: "Once upon a time there was a wolf",
    9: "Once upon a time there was a genie",
    10: "Once upon a time there was a poor boy"
}

generated_stories = [] #generation
output_of_layers = [] #test
token_prompt_ids = [] #analysis

with open("story_dataset.json", 'w') as f:
    for prompt_id in prompts:
        #generation - generate stories for each prompt
        for i in range(1000):
            sequences = llamatales_pipeline(
                prompts[prompt_id],
                do_sample = True,
                top_k = 10,
                num_return_sequences = 1,
                max_new_tokens = 512
            )
            generated_story = sequences[0]['generated_text']
            generated_stories.append(generated_story)
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

            output_of_layers.append(outputs.hidden_states)

            #analysis - collect prompt id of each token
            output_id = []
            for token in num_tokens_generated_story:
                output_id.append(prompt_id)
            token_prompt_ids.append(output_id)
