import os
import json
import matplotlib.pyplot as plt

def save_score_dist(score_list,save_path):
    
    plt.hist(score_list, bins=range(11), edgecolor='black')
 
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    
    plt.xlim(0, max(score_list))
    
    plt.savefig(save_path)
    plt.clf()


def construct_dataset(dir_loc,rule_reward_file,all_hypo_file,apply_rule_file):

    with open(dir_loc+"/"+rule_reward_file,'r') as file:
        rule_reward_orig = json.load(file)

    with open(dir_loc+"/"+all_hypo_file,'r') as file:
        all_hypo_list = json.load(file)

    scores_chosen = []
    scores_rejected = []
    sharegpt_pair_list = []
    apply_rule_list = []
    generate_rule_list = []
    
    for rule_reward, all_hypo in zip(rule_reward_orig,all_hypo_list):
        
        prompt = rule_reward['chosen'][0]
        score_rule_list = []
        
        for hypo in all_hypo:
            score_rule_list.append((hypo['Score'],hypo['Rule']))
            
        score_rule_list.sort(reverse=True)
        
        for score_rule_x in score_rule_list:
            
            for score_rule_y in score_rule_list:
                
                if(score_rule_x[0]>(score_rule_y[0])):
    
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

                    isi = prompt["content"].find("Input:")
                    instruction = prompt["content"][:isi]
                    input_content = prompt["content"][isi:]

                    generate_rule_list.append(
                        {
                            "instruction": instruction,
                            "input": input_content,
                            "output": chosen_rule
                        }
                    )

                    generate_rule_list.append(
                        {
                            "instruction": instruction,
                            "input": input_content,
                            "output": rejected_rule
                        }
                    )
    save_score_dist(scores_chosen,dir_loc+"/chosen.png")
    save_score_dist(scores_rejected,dir_loc+"/rejected.png")

    with open(dir_loc+"/"+apply_rule_file,'r') as file:
        apply_prompt_response_list = json.load(file)

    for prompt_response in apply_prompt_response_list:
        prompt = prompt_response["prompt"]
        response = prompt_response["response"]
        isi = prompt.find("Input:")
        instruction = prompt[:isi]
        input_content = prompt[isi:]

        apply_rule_list.append(
            {
                "instruction": instruction,
                "input": input_content,
                "output": response
            }
        )

    return sharegpt_pair_list, generate_rule_list, apply_rule_list

task_data_locs = [
    (
    "./data/1d_arc/gpt-4",
    "rule_reward_set_train_5_1.0.json",
    "all_hypo_train_5_1.0.json",
    "rule_apply_train_5_1.0.json"
    ),
    (
    "./data/acre/gpt-4",
    "rule_reward_set_train_10_1.0.json",
    "all_hypo_train_10_1.0.json",
    "rule_apply_train_10_1.0.json"
    ),
    (
    "./data/list_func/gpt-4",
    "rule_reward_set_train_10_1.0.json",
    "all_hypo_train_10_1.0.json",
    "rule_apply_train_10_1.0.json"
    ),
    (
    "./data/scan/gpt-4",
    "rule_reward_set_train_10_1.0.json",
    "all_hypo_train_10_1.0.json",
    "rule_apply_train_10_1.0.json"
    )
]

sharegpt_pair_list = []
generate_rule_list = []
apply_rule_list = []

for task_data in task_data_locs:

    spl, grl, arl = construct_dataset(task_data[0],task_data[1],task_data[2],task_data[3])
    sharegpt_pair_list += spl
    generate_rule_list += grl
    apply_rule_list += arl

if(not os.path.exists("./data/merged")):
    os.makedirs("./data/merged")

with open("data/merged/generate_rule_dpo.json",'w') as file:
    json.dump(sharegpt_pair_list,file,indent=4)

with open("data/merged/generate_rule_sft.json",'w') as file:
    json.dump(generate_rule_list,file,indent=4)

with open("data/merged/apply_rule_sft.json",'w') as file:
    json.dump(apply_rule_list,file,indent=4)
                    
            
