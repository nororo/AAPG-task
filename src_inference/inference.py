
"""
Inference script for the model.

"temperature" is miss spelling in the function "temperture", sorry.
"""

from datasets import load_dataset,Dataset

import torch
from torch import cuda, bfloat16
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging
)
from datasets import load_dataset
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
from huggingface_hub import login
import wandb
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import csv

def fet_dataset_inf(filename:str)->pd.DataFrame:
    data_inf_df = pd.read_csv(filename).rename(columns={'output':'audit_res_md'})
    return data_inf_df



def get_pretrained_model(model_name:str,instruct_bool:bool=True,qlora_bool:bool=True):
    if instruct_bool:
        model_name_dict = {
            'Qwen':"Qwen/Qwen2-7B-Instruct",
            'Swallow':"tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.1",
            'Llama':"meta-llama/Meta-Llama-3.1-8B-Instruct"
            }
    else:
        model_name_base_dict = {
            'Qwen':"Qwen/Qwen2-7B",
            'Swallow':"tokyotech-llm/Llama-3.1-Swallow-8B-v0.1",
            'Llama':"meta-llama/Meta-Llama-3.1-8B"
            }
    if model_name in model_name_dict.keys():
        model_name_path = model_name_dict[model_name]
    else:
        ValueError("model_name is not in the list")

    if qlora_bool:
        bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name_path,
            cache_dir=CASHE_DIR,
            trust_remote_code=True,
            token=HUGGINGFACE_TOKEN,
            quantization_config=bnb_config,
            device_map='auto',
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2"
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=CASHE_DIR,
            trust_remote_code=True,
            token=HUGGINGFACE_TOKEN,
            device_map='auto',
            attn_implementation="flash_attention_2"
        )
    if model_name == "Llama":
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_path,
            cache_dir=CASHE_DIR,
            token=token,
            padding_side="right",
            add_eos_token=True
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
    elif model_name == "Qwen":
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_path,
            cache_dir=CASHE_DIR,
            trust_remote_code=True,
            token=HUGGINGFACE_TOKEN,
            padding_side="left",
            add_eos_token=True
        )
        tokenizer.pad_token = "<|im_end|>"
        tokenizer.pad_token_id = 151645

    elif model_name == "Swallow":
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_path,
            cache_dir=CASHE_DIR,
            trust_remote_code=True,
            token=HUGGINGFACE_TOKEN,
            padding_side="right",
            add_eos_token=True
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

    else:
        ValueError("model_name is not in the list")

    return model, tokenizer

def main():
    print("Start training")
    train_model()
    print("Finish training")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_path_str", type=str, default="./cfg/cfg_qwen2_sft_it.yaml")
    parser.add_argument("--output_filename", type=str, default="./results/qwen2_7b/qwen2_instruct")
    parser.add_argument("--filename_eval_data", type=str, default="../dataset/gen_audres_1shot_llama_3.1_8b.jsonl")
    parser.add_argument("--inf_mode", type=str, default="zero-shot", choices=["zero-shot","few-shot"])
    return parser.parse_args()


def inference_few():
    args = parse_args()
    cfg_path_str = args.cfg_path_str
    file_path = Path(cfg_path_str)
    # laod cfg
    cfg_vis = yaml.safe_load(file_path.read_text())
    #sftcfg = SFTConfig({**cfg_vis['cfg_model'],**cfg_vis['cfg_lora'],**cfg_vis['cfg_training_arguments']})
    infcfg = InferenceConfig({**cfg_vis['cfg_model'],**cfg_vis['cfg_inference']})

    model, tokenizer = get_pretrained_model(infcfg.model_name,instruct_bool=sftcfg.instruct_bool,qlora_bool=sftcfg.qlora_bool)

    if infcfg.pre_sft_type:
        # Both for SFT-IT and SFT-CV, adapter is merged on instruction tuned model
        adapter = infcfg['adapter_checkpoint_path']
        model = PeftModel.from_pretrained(model,adapter)
    model.eval()


    if infcfg.inf_mode == "zero-shot":
        data_inf_df = get_dataset_inf(args.filename_eval_data)
        default_system_prompt_en = """As an auditor, you have been given the following audit consideration. Please plan the corresponding audit responses in Japanese."""
        default_system_prompt = """監査担当者であるあなたは、次の監査上の検討事項を与えられました。これに対する監査上の対応事項を日本語文章で具体的に立案してください。"""
        
        with open(args.output_filename, mode='w', encoding='utf-8', newline='') as csv_file:
            fieldnames = ['index_num','description','audit_res','output','id']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()

            for itr in tqdm(data_inf_df.index):
                description = data_inf_df.loc[itr,:].description
                audit_res = data_inf_df.loc[itr,:].audit_res
                id = data_inf_df.loc[itr,:].id
                messages = [
                    {"role": "system", "content": default_system_prompt},
                    {"role": "user", "content": description},
                ]
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                token_ids = tokenizer.encode(
                    prompt, add_special_tokens=False, return_tensors="pt"
                )
                with torch.no_grad():
                    output_ids = model.generate(
                        token_ids.to(model.device),
                        max_new_tokens=infcfg.max_new_tokens,
                        do_sample=infcfg.do_sample,
                        temperature=infcfg.temperture,
                        top_p=infcfg.top_p,
                        pad_token_id=tokenizer.eos_token_id
                    )
                output = tokenizer.decode(
                    output_ids.tolist()[0][token_ids.size(1):], skip_special_tokens=True
                )
                ans_t={'index_num':itr,'description':description,'audit_res':audit_res,'output':output,'id':id}
                writer.writerow(ans_t)

    elif args.inf_mode == "few-shot":
        with open(args.output_filename, mode='w', encoding='utf-8', newline='') as csv_file:
            fieldnames = [
                'index_num',
                'output',
                'api_status',
                'pred',
                'status'
            ]
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            batch_input_obj=batch_input(args.filename_eval_data)

            for index_num in tqdm(batch_input_obj.get_index()):

                sys_prompt,usr_prompt,model_name,max_tokens,temperature=batch_input_obj.get_slice(index_num)
                output=inference_model(
                    model,
                    sys_prompt,
                    usr_prompt,
                    max_tokens
                )
                ans_t={
                        'index_num':index_num,
                        'output':'-',
                        'api_status':'-',
                        'pred':'-',
                        'status':'-',
                        }
                ans_t["output"]=output
                writer.writerow(ans_t)
    else:
        ValueError("inf_mode is not in the list")


class batch_input():
    def __init__(self,filename_openai_input):
        self.batch_input_df = pd.read_json(filename_openai_input, orient='records', lines=True)
    def get_slice(self,index_num):
        sys_prompt=self.batch_input_df.loc[index_num,'body']['messages'][0]['content']
        usr_prompt=self.batch_input_df.loc[index_num,'body']['messages'][1]['content']
        model_name=self.batch_input_df.loc[index_num,'body']['model']
        max_tokens=self.batch_input_df.loc[index_num,'body']['max_tokens']
        temperature=self.batch_input_df.loc[index_num,'body']['temperature']
        return sys_prompt,usr_prompt,model_name,max_tokens,temperature
    def get_index(self):
        return self.batch_input_df.index


def inference_model(model,sys_prompt,usr_prompt,max_tokens):
    messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": usr_prompt},
        ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    token_ids = tokenizer.encode(
        prompt, add_special_tokens=False, return_tensors="pt"
    )
    
    with torch.no_grad():
        output_ids = model.generate(
            token_ids.to(model.device),
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.6,
            #temperature=0.9,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
    output = tokenizer.decode(
        output_ids.tolist()[0][token_ids.size(1):], skip_special_tokens=True
    )
    return output