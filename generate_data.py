import json
import re
from tqdm.auto import tqdm
import argparse
from llm import ChatGPT

parser = argparse.ArgumentParser(description='Run LLM with specific parameters.')
parser.add_argument('--llm_name', type=str, default='gpt-3.5-turbo-1106', help='Name of the language model')
parser.add_argument('--sample_size', type=int, default=50, help='Sample size for rule generation')
parser.add_argument('--temperature', type=float, default=1.0, help='Temperature setting for text generation')

args = parser.parse_args()

llm_name = args.llm_name
sample_size = args.sample_size
temperature = args.temperature

gpt3 = ChatGPT(llm_name)

def extract_list_substring(input_string):

    pattern = r'\[.*?\]'
    match = re.search(pattern, input_string)
    if match:
        return match.group(0)
    return None

if __name__ == '__main__':

    data = []
    with open("./data/list_function.jsonl",'r') as infile:
        for line in infile:
            data.append(eval(line.strip()))
    
    with open("./config/prompts.json",'r') as infile:
        prompts = json.load(infile)
    
    num_test = 0
    num_corr = 0
    
    
    rule_reward_list = []
    all_hypo_list = []
    
    for di,datum in tqdm(enumerate(data)):
    
        rule_reward = {}
        score_list = []
        rule_list = []
        curr_hypo = []
    
        prompt = prompts["lead_prompt"]
        
        for example in datum['train']:
            prompt += "\n"
            prompt += "Input: "+ example['input']+"\n"
            prompt += "Output: "+ example['output']+"\n"
        
        prompt += "\n"
        prompt += prompts["generate_rule"]
        prompt += "\n"
    
        for ri in tqdm(range(sample_size)):
    
            score = 0
            try:
                rule = gpt3.generate(prompt, temperature=temperature)
                
                for ei,example in enumerate(datum['test']):
                    test_prompt = prompts["apply_rule"]+"\n"+rule+"\n"
                    test_prompt = test_prompt+ "Input: "+ example['input']+"\n"
                    response = gpt3.generate(test_prompt, temperature=temperature)
                    prediction = extract_list_substring(response)
                    num_test+=1
                    if(prediction is not None and prediction==example['output']):
                        num_corr+=1
                        score+=1
                    data[di]['test'][ei]["ir-rule"] = rule
                    data[di]['test'][ei]["ir-output"] = prediction
            except:
                print("exception with LLM output")
    
            score_list.append(score)
            rule_list.append(rule)
            curr_hypo.append(
                {
                    "Rule": rule,
                    "Score": score
                }
            )
    
        sorted_list = list(zip(score_list,rule_list))
        sorted_list.sort(reverse=True)
    
        rule_reward['chosen'] = [
            {
                "content": prompt,
                "role": "user"
            },
            {
                "content": sorted_list[0][1],
                "role": "assistant"
            }]
    
        rule_reward['rejected'] = [
            {
                "content": prompt,
                "role": "user"
            },
            {
                "content": sorted_list[-1][1],
                "role": "assistant"
            }]
    
        rule_reward['score_chosen'] = sorted_list[0][0]
        rule_reward['score_rejected'] = sorted_list[-1][0]
        print(sorted_list[0][0],sorted_list[-1][0])
    
        rule_reward_list.append(rule_reward)
        all_hypo_list.append(curr_hypo)
    
        with open("./data/"+"rule_reward_set_"+str(sample_size)+"_"+str(temperature)+".json",'w') as outfile:
            json.dump(rule_reward_list,outfile,indent=4)   
        with open("./data/"+"all_hypo_"+str(sample_size)+"_"+str(temperature)+".json",'w') as outfile:
            json.dump(all_hypo_list,outfile,indent=4)
    
    with open("./data/"+"rule_reward_set_"+str(sample_size)+"_"+str(temperature)+".json",'w') as outfile:
        json.dump(rule_reward_list,outfile,indent=4)

    with open("./data/"+"all_hypo_"+str(sample_size)+"_"+str(temperature)+".json",'w') as outfile:
        json.dump(all_hypo_list,outfile,indent=4)
