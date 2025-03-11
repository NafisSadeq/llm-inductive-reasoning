import os
import json
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

def save_score_dist(score_list,save_path):
    
    plt.hist(score_list, bins=range(11), edgecolor='black')
 
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    
    plt.xlim(0, max(score_list+[5])+1)
    
    plt.savefig(save_path)
    plt.clf()

def construct_dataset(dir_loc,rule_reward_file,all_hypo_file,apply_rule_file,score_diff):

    with open(dir_loc+"/"+rule_reward_file,'r') as file:
        rule_reward_orig = json.load(file)

    with open(dir_loc+"/"+all_hypo_file,'r') as file:
        all_hypo_list = json.load(file)

    scores_chosen = []
    scores_rejected = []
    sharegpt_pair_list = []
    kto_list = []
    apply_rule_list = []
    generate_rule_list = []
    high_quality_rules = set()
    unique_rules = set()
    
    for rule_reward, all_hypo in zip(rule_reward_orig,all_hypo_list):
        
        prompt = rule_reward['chosen'][0]
        score_rule_list = []
        
        for hypo in all_hypo:
            if hypo['Rule'] not in unique_rules:
                unique_rules.add(hypo['Rule'])
                score_rule_list.append((hypo['Score'],hypo['Rule']))
            else:
                continue
            
        score_rule_list.sort(reverse=True)
        kto_count = 0
        
        for score_rule_x in score_rule_list:

            isi = prompt["content"].find("Input:")
            instruction = prompt["content"][:isi]
            input_content = prompt["content"][isi:]

            generate_rule_list.append(
                {
                    "instruction": instruction,
                    "input": input_content,
                    "output": score_rule_x[1]
                }
            )

            if(2*kto_count<len(score_rule_list)):
                kto_label = True
            else:
                kto_label = False

            kto_list.append(
                {
                "messages": [
                  {
                    "content": prompt["content"],
                    "role": "user"
                  },
                  {
                    "content": score_rule_x[1],
                    "role": "assistant"
                  }
                ],
                "label": kto_label
                }
            )
            kto_count+=1
            
            for score_rule_y in score_rule_list:
                
                if(score_rule_x[0]>(score_rule_y[0]+score_diff)):
    
                    sharegpt_pair = {}
                    chosen_rule = score_rule_x[1]
                    chosen_score = score_rule_x[0]
                    rejected_rule = score_rule_y[1]
                    rejected_score = score_rule_y[0]
                
                    scores_chosen.append(chosen_score)
                    scores_rejected.append(rejected_score)
    
                    sharegpt_pair["conversations"] = [
                        {
                            "from": "human",
                            "value": prompt["content"]
                        }
                    ]
    
                    sharegpt_pair["chosen"] = {
                        "from": "gpt",
                        "value": chosen_rule
                    }
    
                    sharegpt_pair["rejected"] = {
                        "from": "gpt",
                        "value": rejected_rule
                    }
    
                    sharegpt_pair_list.append(sharegpt_pair)
                    high_quality_rules.add(chosen_rule)

    save_score_dist(scores_chosen,dir_loc+"/chosen.png")
    save_score_dist(scores_rejected,dir_loc+"/rejected.png")

    with open(dir_loc+"/"+apply_rule_file,'r') as file:
        apply_prompt_response_list = json.load(file)

    for prompt_response in tqdm(apply_prompt_response_list):
        prompt = prompt_response["prompt"]
        response = prompt_response["response"]
        isi = prompt.find("Input:")
        instruction = prompt[:isi]
        input_content = prompt[isi:]

        for rule in high_quality_rules:

            if(rule in prompt):
                apply_rule_list.append(
                    {
                        "instruction": instruction,
                        "input": input_content,
                        "output": response
                    }
                )
                break

    print(dir_loc,len(sharegpt_pair_list),len(kto_list),len(generate_rule_list),len(apply_rule_list))

    return sharegpt_pair_list, kto_list, generate_rule_list, apply_rule_list

task_data_locs = [
    (
    "./data/1d_arc/gpt-4",
    "rule_reward_set_train_25_1.0.json",
    "all_hypo_train_25_1.0.json",
    "rule_apply_train_25_1.0.json",
    1
    ),
    (
    "./data/acre/gpt-4",
    "rule_reward_set_train_50_1.0.json",
    "all_hypo_train_50_1.0.json",
    "rule_apply_train_50_1.0.json",
    2
    ),
    (
    "./data/list_func/gpt-4",
    "rule_reward_set_train_50_1.0.json",
    "all_hypo_train_50_1.0.json",
    "rule_apply_train_50_1.0.json",
    3
    ),
    (
    "./data/scan/gpt-4",
    "rule_reward_set_train_50_1.0.json",
    "all_hypo_train_50_1.0.json",
    "rule_apply_train_50_1.0.json",
    4
    )
]

sharegpt_pair_list = []
kto_list = []
generate_rule_list = []
apply_rule_list = []

for task_data in task_data_locs:

    spl, kl, grl, arl = construct_dataset(task_data[0],task_data[1],task_data[2],task_data[3],task_data[4])
    sharegpt_pair_list += spl
    kto_list += kl
    generate_rule_list += grl
    apply_rule_list += arl

if(not os.path.exists("./data/merged")):
    os.makedirs("./data/merged")

with open("data/merged/generate_rule_dpo.json",'w') as file:
    json.dump(sharegpt_pair_list,file,indent=4)

with open("data/merged/generate_rule_kto.json",'w') as file:
    json.dump(kto_list,file,indent=4)

with open("data/merged/generate_rule_sft.json",'w') as file:
    json.dump(generate_rule_list,file,indent=4)

with open("data/merged/apply_rule_sft.json",'w') as file:
    json.dump(apply_rule_list,file,indent=4)

few_shot_sft_files = [
    "./data/list_func/gpt-4/fewshot_io_sft.json",
    "./data/1d_arc/gpt-4/fewshot_io_sft.json",
    "./data/acre/gpt-4/fewshot_io_sft.json",
    "./data/scan/gpt-4/fewshot_io_sft.json"
]

few_shot_sft = []

for file_path in few_shot_sft_files:
    with open(file_path,'r') as infile:
        content = json.load(infile)
        print(file_path,len(content))
        few_shot_sft += content

with open("data/merged/fewshot_io_sft.json",'w') as file:
    json.dump(few_shot_sft,file,indent=4)
        
