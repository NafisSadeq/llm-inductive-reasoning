## Install

Create conda environment using the provided requirements.txt

```
conda create -n <environment-name> --file requirements.txt
```


## Data augmentation

Move to code directory

```
cd ./code/
```

Perform data augmentation using a teacher model such GPT-4o

```
python generate_data_ir.py --llm_name gpt-4o --hypo_size 50 --task list_func
python generate_data_ir.py --llm_name gpt-4o --hypo_size 50 --task 1d_arc
python generate_data_ir.py --llm_name gpt-4o --hypo_size 50 --task acre
python generate_data_ir.py --llm_name gpt-4o --hypo_size 50 --task scan
```

Perform data filtering and merge the task-specific data segments

```
python construct_pair.py
```

This step will generate three datasets, generate_rule_sft.json, apply_rule_sft.json, and generate_rule_dpo.json. We already provide the three datasets within ./dataset/ folder

## Model training

We use LLaMA-Factory for model training. Install LLaMA-Factory following instructions here: https://github.com/hiyouga/LLaMA-Factory/tree/main?tab=readme-ov-file#installation. Copy the three dataset files within LLaMA-Factory/data folder and update LLaMA-Factorydata/dataset_info.json as required. Follow the instructions here: https://github.com/hiyouga/LLaMA-Factory/blob/main/data/README.md. Then you can perform suprvised fine-tuning on the generate_rule_sft.json + apply_rule_sft.json datasets as follows. 

```  
llamafactory-cli train examples/train_lora/llama3_lora_sft.yaml
```

After that you can perform alignment on the generate_rule_dpo.json dataset as follows.

```
llamafactory-cli train examples/train_lora/llama3_lora_dpo.yaml
```

Make sure to update the training configuration files llama3_lora_sft.yaml/llama3_lora_dpo.yaml with appropriate model name, dataset names, and hyper-parameters.

## Inference

Load the base models with corresponding LoRA adapters created during model tuning for inference.

```
python proposed.py --task list_func --llm_name meta-llama/Meta-Llama-3-8B-Instruct --adapter_path <adapter_path> --hypo_size 10 --rg_temp 0.9 --rf_temp 0.7
```

We also provide codes for corresponding baseline codes for both direct few-shot prompting and hypothesis search. For direct few-shot prompting, you can run

```
python baseline_io.py --task list_func --llm_name meta-llama/Meta-Llama-3-8B-Instruct
```

For hypothesis search with base LLaMA model, you can run

```
python baseline_ir.py --task list_func --llm_name meta-llama/Meta-Llama-3-8B-Instruct --hypo_size 10
```