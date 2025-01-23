import os
import json
import random
import argparse
from tqdm.auto import tqdm
from utils import extract_list_substring
from llm import ChatGPT, LlamaAdapter, QwenAdapter

parser = argparse.ArgumentParser(description='Run LLM with specific parameters.')
parser.add_argument('--llm_name', type=str, default="meta-llama/Meta-Llama-3-8B-Instruct",choices=[
    'meta-llama/Meta-Llama-3-8B-Instruct',
    'mistralai/Mistral-7B-Instruct-v0.3', 
    'Qwen/Qwen2.5-7B-Instruct',
    'gpt-3.5-turbo-1106',
    'gpt-4o'
], help='Name of the language model')
parser.add_argument('--task', type=str, default="list_func",choices=[
    'list_func',
    '1d_arc', 
    'acre', 
    'scan'
], help='Task Name')
parser.add_argument('--adapter_path', type=str, required = False, help='Name or path of Lora adapter')

args = parser.parse_args()
random.seed(10)

llm_name = args.llm_name
if(args.adapter_path):
    adapter_name = args.adapter_path
else:
    adapter_name = None
task = args.task

if(llm_name.startswith("Qwen")):
    llm = QwenAdapter(llm_name,adapter_name)
    llm_tag = "qwen"
elif(llm_name.startswith("meta")):
    llm = LlamaAdapter(llm_name,adapter_name)
    llm_tag = "llama3"
elif(llm_name.startswith("mistralai")):
    llm = LlamaAdapter(llm_name,adapter_name)
    llm_tag = "mistral"
else:
    llm = ChatGPT(llm_name)
    llm_tag = llm_name[:5]

if(task == "list_func"):
    data_path = "./data/list_func/list_function.jsonl"
elif(task == "1d_arc"):
    data_path = "./data/1d_arc/1D_arc.jsonl"
elif(task == "acre"):
    data_path = "./data/acre/acre.jsonl"
elif(task == "scan"):
    data_path = "./data/scan/scan.jsonl"
else:
    data_path = None

data = []
with open(data_path,'r') as infile:
    for line in infile:
        data.append(eval(line.strip()))

random.shuffle(data)
train_len = int(len(data)*0.9)
data = data[train_len:]

with open("./config/prompts.json",'r') as infile:
    prompts = json.load(infile)

num_test = 0
num_corr = 0

print("# sample",len(data))

for di,datum in enumerate(tqdm(data)):

    prompt = prompts[task]["direct_fewshot1"]
    
    for example in datum['train']:
        prompt += "\n"
        prompt += "Input: "+ str(example['input'])+"\n"
        prompt += "Output: "+ str(example['output'])+"\n"
    
    prompt += "\n"
    
    for ei,example in enumerate(datum['test']):
        test_prompt = prompt + prompts[task]["direct_fewshot2"]+"\n"
        test_prompt = test_prompt+ "Input: "+ str(example['input'])+"\n"
        response = llm.generate(test_prompt)
        if(task=="list_func" or task=="1d_arc"):
            prediction = extract_list_substring(response)
        else:
            prediction = response
        #print("Ground truth:",example['output'])
        #print("Prediction:",prediction)
        num_test+=1
        if(prediction is not None and prediction==example['output']):
            num_corr+=1
        data[di]['test'][ei]["ir-output"] = prediction

print("Accuracy:",round(num_corr/num_test,2))

output_dir = "./outputs/"+task+"/"+llm_tag

if(not os.path.exists(output_dir)):
    os.makedirs(output_dir)
    
# with open(output_dir+"/baseline_ir_prompt.json",'w') as outfile:
#     json.dump(data,outfile,indent=4)
