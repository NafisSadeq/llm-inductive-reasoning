import json
from llm import ChatGPT,LLAMA,LlamaAdapter
import re

llm_name = "meta-llama/Meta-Llama-3-8B-Instruct"
adapter_name = "llama-3-dpo/checkpoint-8436"

def extract_list_substring(input_string):

    pattern = r'\[.*?\]'
    match = re.search(pattern, input_string)
    if match:
        return match.group(0)
    return None

llm = LlamaAdapter(llm_name,adapter_name)

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

with open("./outputs/"+"proposed.json",'w') as outfile:
    json.dump(data,outfile,indent=4)