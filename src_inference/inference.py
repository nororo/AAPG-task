"""Inference script for the model.

# Example: sft-it
python inference_milk.py --cfg_path_str ../cfg/milk/cfg_inf_qwen2_it.yaml --output_filename ../../results_milk/qwen2_7b/qwen2_instruct/inf_qwen2_sft_it_7b.csv --filename_eval_data ../../data/milk/batch_gen_ans_llama_3.1_8b.jsonl

# Example: sft cv
python inference_milk.py --cfg_path_str ../cfg/milk/cfg_inf_qwen2_cv.yaml --output_filename ../../results_milk/qwen2_7b/qwen2_merge/inf_qwen2_sft_cv_7b.csv --filename_eval_data ../../data/milk/batch_gen_ans_llama_3.1_8b.jsonl


# Example: plain
python inference_milk.py --cfg_path_str ../cfg/milk/cfg_inf_qwen2_icl.yaml --output_filename ../../results_milk/qwen2_7b/qwen2_instruct/inf_qwen2_plain.csv --filename_eval_data ../../data/milk/batch_gen_ans_llama_3.1_8b.jsonl

# Example: icl knn 20
python inference_milk.py --cfg_path_str ../cfg/milk/cfg_inf_qwen2_icl.yaml --output_filename ../../results_milk/qwen2_7b/qwen2_instruct/inf_qwen2_knn_icl.csv --filename_eval_data ../../data/milk/batch_gen_ans_knn_1_20_llama_3.1_8b.jsonl

"""

import argparse
import csv
from pathlib import Path

import pandas as pd
import torch
import yaml
from peft import PeftModel
from pydantic import BaseModel, Field
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

CASHE_DIR = "your_cache_dir"
HUGGINGFACE_TOKEN = "your_huggingface_token"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg_path_str",
        type=str,
        default="./cfg/cfg_inf_qwen2_it.yaml",
    )
    parser.add_argument(
        "--output_filename",
        type=str,
        default="./results/qwen2_7b/qwen2_instruct",
    )
    parser.add_argument(
        "--filename_eval_data",
        type=str,
        default="../dataset/gen_audres_1shot_llama_3.1_8b.jsonl",
    )
    return parser.parse_args()


class InferenceConfig(BaseModel):
    pre_model_name: str = Field(
        isin=["Qwen", "Swallow", "Llama"],
        title="pretrained model name",
    )
    pre_qlora_bool: bool = Field(default=True, title="whether the model is qlora model")
    pre_instruct_bool: bool = Field(
        default=True,
        title="whether the model is instruct model",
    )
    pre_sft_bool: bool = Field(default=False, title="whether the model is sft model")
    pre_adapter_checkpoint_path: str = Field(
        default="",
        title="adapter checkpoint path",
    )

    input_file_type: str = Field(default="openai_batch", title="input file type")
    max_new_tokens: int = Field(default=1024, title="max new tokens")
    do_sample: bool = Field(default=True, title="do sample")
    temperature: float = Field(default=0.6, title="temperature")
    top_p: float = Field(default=0.9, title="top p")


def get_dataset_inf(filename: str) -> pd.DataFrame:
    data_inf_df = pd.read_csv(filename).rename(columns={"output": "audit_res_md"})
    return data_inf_df


def get_dataset_inf_med(filename: str) -> pd.DataFrame:
    data_inf_df = pd.read_csv(filename)
    return data_inf_df


def get_pretrained_model(
    model_name: str,
    instruct_bool: bool = True,
    qlora_bool: bool = True,
):
    if instruct_bool:
        model_name_dict = {
            "Qwen": "Qwen/Qwen2-7B-Instruct",
            "Swallow": "tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.1",
            "Llama": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        }
    else:
        model_name_dict = {
            "Qwen": "Qwen/Qwen2-7B",
            "Swallow": "tokyotech-llm/Llama-3.1-Swallow-8B-v0.1",
            "Llama": "meta-llama/Meta-Llama-3.1-8B",
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
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name_path,
            cache_dir=CASHE_DIR,
            trust_remote_code=True,
            token=HUGGINGFACE_TOKEN,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_path,
            cache_dir=CASHE_DIR,
            trust_remote_code=True,
            token=HUGGINGFACE_TOKEN,
            torch_dtype=torch.float16,
            device_map="auto",
        )
    if model_name == "Llama":
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_path,
            cache_dir=CASHE_DIR,
            token=HUGGINGFACE_TOKEN,
            padding_side="right",
            add_eos_token=True,
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
            add_eos_token=True,
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
            add_eos_token=True,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

    else:
        ValueError("model_name is not in the list")

    return model, tokenizer


def main():
    print("Start inference")
    inference_few()
    print("Finish inference")


def inference_few():
    args = parse_args()
    cfg_path_str = args.cfg_path_str
    file_path = Path(cfg_path_str)
    # laod cfg
    cfg_vis = yaml.safe_load(file_path.read_text())
    infcfg = InferenceConfig(**{**cfg_vis["cfg_model"], **cfg_vis["cfg_inference"]})

    model, tokenizer = get_pretrained_model(
        infcfg.pre_model_name,
        instruct_bool=infcfg.pre_instruct_bool,
        qlora_bool=infcfg.pre_qlora_bool,
    )

    if infcfg.pre_sft_bool:
        # Both for SFT-IT and SFT-CV, adapter is merged on instruction tuned model
        adapter = infcfg.pre_adapter_checkpoint_path
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()

    if infcfg.input_file_type != "openai_batch":
        data_inf_df = get_dataset_inf(args.filename_eval_data)
        default_system_prompt = cfg_vis["dataset"]["default_system_prompt"]

        with open(
            args.output_filename,
            mode="w",
            encoding="utf-8",
            newline="",
        ) as csv_file:
            fieldnames = ["index_num", "description", "audit_res", "output", "id"]
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()

            for itr in tqdm(data_inf_df.index):
                description = data_inf_df.loc[itr, :].description
                audit_res = data_inf_df.loc[itr, :].audit_res
                id = data_inf_df.loc[itr, :].id
                messages = [
                    {"role": "system", "content": default_system_prompt},
                    {"role": "user", "content": description},
                ]
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                token_ids = tokenizer.encode(
                    prompt,
                    add_special_tokens=False,
                    return_tensors="pt",
                )
                with torch.no_grad():
                    output_ids = model.generate(
                        token_ids.to(model.device),
                        max_new_tokens=infcfg.max_new_tokens,
                        do_sample=infcfg.do_sample,
                        temperature=infcfg.temperature,
                        top_p=infcfg.top_p,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                output = tokenizer.decode(
                    output_ids.tolist()[0][token_ids.size(1) :],
                    skip_special_tokens=True,
                )
                ans_t = {
                    "index_num": itr,
                    "description": description,
                    "audit_res": audit_res,
                    "output": output,
                    "id": id,
                }
                writer.writerow(ans_t)

    elif infcfg.input_file_type == "openai_batch":
        with open(
            args.output_filename,
            mode="w",
            encoding="utf-8",
            newline="",
        ) as csv_file:
            fieldnames = [
                "index_num",
                "output",
                "api_status",
                "pred",
                "status",
            ]
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            batch_input_obj = batch_input(args.filename_eval_data)

            for index_num in tqdm(batch_input_obj.get_index()):
                sys_prompt, usr_prompt, max_tokens, temperature = (
                    batch_input_obj.get_slice(index_num)
                )
                output = inference_model(
                    model,
                    sys_prompt,
                    usr_prompt,
                    max_tokens,
                    temperature,
                    tokenizer,
                )
                ans_t = {
                    "index_num": index_num,
                    "output": "-",
                    "api_status": "-",
                    "pred": "-",
                    "status": "-",
                }
                ans_t["output"] = output
                writer.writerow(ans_t)
    else:
        ValueError("inf_mode is not in the list")


class batch_input:
    def __init__(self, filename_openai_input):
        self.batch_input_df = pd.read_json(
            filename_openai_input,
            orient="records",
            lines=True,
        )

    def get_slice(self, index_num):
        sys_prompt = self.batch_input_df.loc[index_num, "body"]["messages"][0][
            "content"
        ]
        usr_prompt = self.batch_input_df.loc[index_num, "body"]["messages"][1][
            "content"
        ]
        max_tokens = self.batch_input_df.loc[index_num, "body"]["max_tokens"]
        temperature = self.batch_input_df.loc[index_num, "body"]["temperature"]
        return sys_prompt, usr_prompt, max_tokens, temperature

    def get_index(self):
        return self.batch_input_df.index


def inference_model(model, sys_prompt, usr_prompt, max_tokens, temperature, tokenizer):
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": usr_prompt},
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    token_ids = tokenizer.encode(
        prompt,
        add_special_tokens=False,
        return_tensors="pt",
    )

    with torch.no_grad():
        output_ids = model.generate(
            token_ids.to(model.device),
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    output = tokenizer.decode(
        output_ids.tolist()[0][token_ids.size(1) :],
        skip_special_tokens=True,
    )
    return output


if __name__ == "__main__":
    main()
