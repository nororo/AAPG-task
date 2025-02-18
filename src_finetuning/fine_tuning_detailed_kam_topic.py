
"""
This is for Supervised Fine-tuning (Detail-SFT-IT and Detail-SFT-CV) in Appendix, specialized for each KAM topic cluster (mixture of adapters).

"""

from datasets import load_dataset,Dataset

import transformers
assert transformers.__version__ == '4.44.2'

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
import numpy as np
import os
import argparse
from pathlib import Path
import yaml

from pydantic import BaseModel, Field,SecretStr
from transformers import DataCollatorForLanguageModeling
from trl import DataCollatorForCompletionOnlyLM
from transformers import DataCollatorWithPadding, Trainer, TrainingArguments


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_path_str", type=str, default="./cfg/cfg_qwen2_sft_it.yaml")
    parser.add_argument("--model_save_dir", type=str, default="./results/qwen2_7b/qwen2_instruct")
    parser.add_argument("--filename_train_data", type=str, default="./data/audit_data.csv")

    parser.add_argument("--checkpoint_path_str", type=str, default="../results/llama_8b_ppo/train_sft_cls/start_checkpoint", help="THE COPY of checkpoint path containing the weight of the model (adapter.safetensor), training status (train_state), training arguments (training_args.bin), and tokenizer arguments (tokenizer, tokenizer_config and special_token_map)")
    
    return parser.parse_args()

class SFTConfig(BaseModel):
    pre_model_name: str = Field(isin=['Qwen','Swallow','Llama'], title="pretrained model name")
    pre_instruct_bool: bool = Field(default=True, title="whether the model is instruct model")
    pre_qlora_bool: bool = Field(default=True, title="whether the model is qlora model")
    
    lora_r: int = Field(default=8, title="lora r")
    lora_alpha: int = Field(default=16, title="lora alpha")
    lora_dropout: float = Field(default=0.05, title="lora dropout")
    lora_target_modules: list = Field(default=["q_proj", "k_proj", "v_proj", "o_proj","gate_proj", "up_proj", "down_proj",], title="target modules")
    lora_bias: str = Field(default="none", title="bias")
    lora_task_type: str = Field(default="CAUSAL_LM", title="task type")
    
    ta_output_dir: str = Field(default="./results/qwen2_7b/qwen2_instruct", title="output dir")
    ta_overwrite_output_dir: bool = Field(default=True, title="overwrite output dir")
    ta_per_device_train_batch_size: int = Field(default=2, title="per device train batch size")
    ta_per_device_eval_batch_size: int = Field(default=4, title="per device eval batch size")
    ta_gradient_accumulation_steps: int = Field(default=1, title="gradient accumulation steps")
    ta_eval_accumulation_steps: int = Field(default=1, title="eval accumulation steps")
    ta_learning_rate: float = Field(default=2e-5, title="learning rate")
    ta_weight_decay: float = Field(default=0.01, title="weight decay")
    ta_num_train_epochs: int = Field(default=6, title="num train epochs")
    ta_evaluation_strategy: str = Field(default='steps', title="evaluation strategy")
    ta_eval_steps: int = Field(default=200, title="eval steps")
    ta_logging_steps: int = Field(default=200, title="logging steps")
    ta_lr_scheduler_type: str = Field(default="constant", title="lr scheduler type")
    ta_warmup_steps: int = Field(default=0, title="warmup steps")
    ta_save_strategy: str = Field(default="epoch", title="save strategy")
    ta_save_steps: int = Field(default=1, title="save steps")
    ta_fp16: bool = Field(default=True, title="fp16")
    ta_report_to: str = Field(default="wandb", title="report to")
    ta_optim: str = Field(default="adamw_8bit", title="optim")
    max_seq_length: int = Field(default=1024, title="max seq length")

def get_dataset_detail(filename:str):
    data_train = pd.read_csv(filename).rename(columns={'output':'audit_res_md'})
    print(data_train.shape) #8350
    
    # cluster label
    filename="../dataset/data_all17k_pivot_2407_v1012_with_cls.csv"
    cls_df=pd.read_csv(filename)[['id','cls_labels']]
    print("Asigned cluster num")
    print(cls_df.cls_labels.value_counts())
    data_train=pd.merge(
        data_train,
        cls_df,
        left_on='id',
        right_on='id',
        how='left')

    dataset = Dataset.from_pandas(data_train)
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    return dataset

def main():
    print("Start training")
    for itr_cls in range(1,13):
        print(f"Training for cluster {itr_cls}")
        train_model(itr_cls)
    print("Finish training")

def train_model(itr_cls:int=1):
    args = parse_args()
    cfg_path_str = args.cfg_path_str
    checkpoint_path = Path(args.checkpoint_path_str)
    file_path = Path(cfg_path_str)
    # laod cfg
    cfg_vis = yaml.safe_load(file_path.read_text())
    sftcfg = SFTConfig({**cfg_vis['cfg_model'],**cfg_vis['cfg_lora'],**cfg_vis['cfg_training_arguments']})

    model_save_dir=Path(args.model_save_dir) / str(itr_cls)
    model_save_dir.mkdir(parents=True,exist_ok=True)
    model, tokenizer = get_pretrained_model(sftcfg.model_name,instruct_bool=sftcfg.instruct_bool,qlora_bool=sftcfg.qlora_bool)
   
    dataset = get_dataset_detail(args.filename_train_data)
    dataset = dataset.map(formatting_prompts_func,batched=True)
    tokenized_dataset = dataset.map(
        lambda sample: tokenizer(sample["text"]), batched=True, remove_columns=list(dataset['train'].features)
    )
    data_collator = get_data_collator()

    peft_config = LoraConfig(
        r=sftcfg.lora_r,
        lora_alpha=sftcfg.lora_alpha,
        lora_dropout=sftcfg.lora_dropout,
        target_modules=sftcfg.lora_target_modules,
        bias=sftcfg.lora_bias,
        task_type=sftcfg.lora_task_type,
        modules_to_save=sftcfg.lora_modules_to_save
        )

    training_args = TrainingArguments(
        output_dir=str(model_save_dir),
        overwrite_output_dir=True,
        per_device_train_batch_size=sftcfg.ta_per_device_train_batch_size,
        per_device_eval_batch_size=sftcfg.ta_per_device_eval_batch_size,
        gradient_accumulation_steps=sftcfg.ta_gradient_accumulation_steps,
        eval_accumulation_steps=sftcfg.ta_eval_accumulation_steps,
        learning_rate=sftcfg.ta_learning_rate,
        weight_decay=sftcfg.ta_weight_decay,
        num_train_epochs=sftcfg.ta_num_train_epochs,
        evaluation_strategy=sftcfg.ta_evaluation_strategy,
        eval_steps=round(len(dataset['train'])/4),
        logging_steps=round(len(dataset['train'])/4),
        lr_scheduler_type=sftcfg.ta_lr_scheduler_type,
        warmup_steps=sftcfg.ta_warmup_steps,
        save_strategy=sftcfg.ta_save_strategy,
        save_steps=sftcfg.ta_save_steps,
        fp16=sftcfg.ta_fp16,
        report_to=sftcfg.ta_report_to,
        optim=sftcfg.ta_optim
    )
    trainer = SFTTrainer(
        model=model,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['test'],
        formatting_func=formatting_prompts_func,
        data_collator=data_collator,
        peft_config=peft_config,
        args=training_args,
        max_seq_length=sftcfg.max_seq_length
        )
    #update check point trainer_state
    with open(checkpoint_path/"trainer_state.json") as f:
        train_state=json.load(f)

    train_state['max_steps']=round((len(dataset['train'])*6)/2)
    train_state['logging_steps']=round(len(dataset['train'])/4)
    train_state['eval_steps']=round(len(dataset['train'])/4)
    train_state['num_train_epochs']=6

    with open(checkpoint_path/"trainer_state.json", mode='wt', encoding='utf-8') as f:
        json.dump(train_state, f , ensure_ascii=False, indent=2)
    
    # train
    trainer.train(resume_from_checkpoint=checkpoint_path)
    trainer.model.save_pretrained(str(model_save_dir/str(itr_cls))) 

    out_filename=str(model_save_dir / str(itr_cls)/ "traing_log_history.json")
    with open(out_filename, mode="wt", encoding="utf-8") as f:
        json.dump(trainer.state.log_history, f, ensure_ascii=False, indent=2)
    
    del model
    del trainer
    torch.cuda.empty_cache()


def login_hf():
    os.system('huggingface-cli login --token $HUGGINGFACE_TOKEN')
    os.system('huggingface-cli whoami')


def get_data_collator():
    tokenizer.pad_token = tokenizer.eos_token
    response_template= "### Response"
    data_collator = DataCollatorForCompletionOnlyLM(response_template=response_template,tokenizer=tokenizer, mlm=False)
    return data_collator


def formatting_prompts_func(examples):
    default_system_prompt_en = """As an auditor, you have been given the following audit consideration. Please plan the corresponding audit responses in Japanese."""
    default_system_prompt = """監査担当者であるあなたは、次の監査上の検討事項を与えられました。これに対する監査上の対応事項を日本語文章で具体的に立案してください。"""
    eos_token = tokenizer.eos_token
    alpaca_prompt = """
    ### Instruction:
    {} 

    ### Input:
    {}

    ### Response:
    {}"""

    instruction = default_system_prompt
    inputs = examples["description"]
    outputs = examples["audit_res_md"]
    texts = []
    for itr_input, output in zip(inputs, outputs):
        text = alpaca_prompt.format(instruction,itr_input, output) + eos_token
        texts.append(text)
    return { "text" : texts, }

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

if __name__ == "__main__":
    main()