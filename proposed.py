import json
from llm import ChatGPT,LLAMA,LlamaAdapter,Qwen, QwenAdapter
import re
import argparse

parser = argparse.ArgumentParser(description='Run LLM with specific parameters.')
parser.add_argument('--llm_name', type=str, default="meta-llama/Meta-Llama-3-8B-Instruct",choices=[
    'meta-llama/Meta-Llama-3-8B-Instruct',
    'mistralai/Mistral-7B-Instruct-v0.3', 
    'Qwen/Qwen2.5-7B-Instruct'
], help='Name of the language model')

args = parser.parse_args()

llm_name = args.llm_name

def extract_list_substring(input_string):

    pattern = r'\[.*?\]'
    match = re.search(pattern, input_string)
    if match:
        return match.group(0)
    return None

if(llm_name.startswith("Qwen")):
    adapter_name = "qwen_dpo"
    llm = QwenAdapter(llm_name,adapter_name)
    llm_tag = "qwen"
elif(llm_name.startswith("meta")):
    adapter_name = "llama-3-dpo/checkpoint-8436"
    llm = LlamaAdapter(llm_name,adapter_name)
    llm_tag = "llama3"
else:
    adapter_name = "mistral_dpo"
    llm = LlamaAdapter(llm_name,adapter_name)
    llm_tag = "mistral"

data = []
with open("./data/list_function.jsonl",'r') as infile:
    for line in infile:
        data.append(eval(line.strip()))

prompt_ir1 = "Consider the following input-output examples.\n"
prompt_ir2 = "Based on the example shown above, identify the rule that can be used to convert the input into output. Your output should solely contain the rule.\n"
prompt_ir3 = "The following problem contains a rule and an input. Use the rule to produce the output from the given input. Your output should solely be a valid python list that represents the output. No other text or description should be present.\n" 

num_test = 0
num_corr = 0

for di,datum in enumerate(data):

    prompt = prompt_ir1
    
    for example in datum['train']:
        prompt += "\n"
        prompt += "Input: "+ example['input']+"\n"
        prompt += "Output: "+ example['output']+"\n"
    
    prompt += "\n"
    prompt += prompt_ir2
    prompt += "\n"

    rule = llm.generate(prompt)
    print(rule)
    
    for ei,example in enumerate(datum['test']):
        test_prompt = prompt_ir3+"\n"+rule+"\n"
        test_prompt = test_prompt+ "Input: "+ example['input']+"\n"
        response = llm.generate(test_prompt)
        prediction = extract_list_substring(response)
        print("Ground truth:",example['output'])
        print("Prediction:",prediction)
        num_test+=1
        if(prediction is not None and prediction==example['output']):
            num_corr+=1
        data[di]['test'][ei]["ir-rule"] = rule
        data[di]['test'][ei]["ir-output"] = prediction

print("Accuracy:",round(num_corr/num_test,2))

with open("./outputs/"+llm_tag+"_proposed.json",'w') as outfile:
    json.dump(data,outfile,indent=4)