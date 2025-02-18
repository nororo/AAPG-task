# %%
from libs.model_api import *
from libs.compose_prompt import *
from libs.utils import *
from libs.kam_evaluation import *
from libs.sudachi_tokenizer import *

import pandas as pd
import numpy as np
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')

import time

#import torch
import csv
import json
import os
from os.path import join, dirname
from yaml import safe_load

from omegaconf import DictConfig
from typing import Optional

from dotenv import load_dotenv
from pathlib import Path
import Levenshtein
import re

import argparse
PROJPATH=r"/Users/noro/Documents/Projects/XBRL_common_space_projection/"
PROJDIR=Path(PROJPATH)

# %%
parser = argparse.ArgumentParser(description='')
parser.add_argument('--out_dir', default=str(PROJDIR / "data/3_processed/dataset_2310/downstream/baseline/gpt_4o_mini"), type=str, help='')
parser.add_argument('--out_ext', default="eval_plane_gpt_4o_mini_", type=str, help='')
parser.add_argument('--input_batch_filename', default=str(PROJDIR / "data/3_processed/dataset_2310/downstream/baseline/gpt_4o_mini" / "aud_res_gpt_4o_mini_output.jsonl")
    , type=str, help='')
parser.add_argument('--model_type', default="openai", type=str, help='')
parser.add_argument('--trial_flg', default=True, type=str, help='')

args = parser.parse_args()
OUT_DIR=args.out_dir
OUT_EXT=args.out_ext
INPUT_FILENAME=args.input_filename
MODEL_TYPE=args.model_type
TRIAL_FLG=args.trial_flg
# %%


# %%
#from openai import OpenAI
load_dotenv(verbose=True)
dotenv_path = join(PROJDIR / "env" / "k", '.env')
load_dotenv(dotenv_path)
openai_api_obj=openai_api()

def make_eval_batch(out_dir,out_ext,input_filename,model_type,trial_flg,model_name_judge="gpt_4o_mini",model_name_extract="gpt_4o_mini"):
    eval_prompt_path=str(PROJDIR / "data/3_processed/dataset_2310/downstream/eval_prep" / "eval_prompt.json")
    eval_output_obj=eval_output(
        prompt_path=eval_prompt_path,
        out_dir=out_dir,
        model_name_extract="gpt_4o_mini",
        model_name_judge=model_name_judge
        )
    filename=PROJDIR / "data/3_processed/dataset_2310/downstream" / "2_intermediate/llm_proc" /"audit_res_markdown_eval.csv"
    if trial_flg:
        data_val_df=pd.read_csv(filename,index_col=None,dtype=str).set_index('index_num').head(100)
    else:
        data_val_df=pd.read_csv(filename,index_col=None,dtype=str).set_index('index_num')

    if model_type == "openai":
        filename_openai_rst=input_filename
        response_list=get_results_openai_batch(filename_openai_rst=filename_openai_rst,json=False)
        proposed_res_list=[item_dict['output'] for item_dict in response_list if item_dict['status']=='Success']
        if trial_flg:
            proposed_res_list=proposed_res_list#[:50]
        eval_output_obj.gen_eval_batch(proposed_res_list,out_filename_without_ext=out_ext)
    elif model_type == "llama":
        inf_df=pd.read_csv(input_filename)
        proposed_res_list=inf_df['output'].to_list()
        if trial_flg:
            proposed_res_list=proposed_res_list#[:50]
        eval_output_obj.gen_eval_batch(proposed_res_list,out_filename_without_ext=out_ext)
    #filename="/Users/noro/Documents/Projects/XBRL_common_space_projection/data/3_processed/dataset_2310/downstream/3_processed/eval_sft7500_jp_addon_instruct.csv"

# %% script
#model_name_judge = "gpt_4o_mini"
model_name_judge = "gpt_4o"
# %%

out_dir = PROJDIR / "data/3_processed/dataset_2310/downstream/baseline/gpt_4"
out_dir.mkdir(parents=True, exist_ok=True)
make_eval_batch(
    out_dir=str(out_dir),
    out_ext="eval_plane_gpt_4_",
    input_filename=str(out_dir / "aud_res_gpt_4_output.jsonl"),
    model_type="openai",
    trial_flg=True,
    model_name_judge=model_name_judge,
    )

# gpt 4 shot

out_dir = PROJDIR / "data/3_processed/dataset_2310/downstream/baseline/gpt_4_1shot"
out_dir.mkdir(parents=True, exist_ok=True)
make_eval_batch(
    out_dir=str(out_dir),
    out_ext="eval_1shot_gpt_4_",
    input_filename=str(out_dir / "batch_gen_audres_1shot_gpt_4_output.jsonl"),
    model_type="openai",
    trial_flg=True,
    model_name_judge=model_name_judge,
    )

# %%
out_dir = PROJDIR / "data/3_processed/dataset_2310/downstream/baseline/gpt_4o"
out_dir.mkdir(parents=True, exist_ok=True)
make_eval_batch(
    out_dir=str(out_dir),
    out_ext="eval_plane_gpt_4o_",
    input_filename=str(out_dir / "aud_res_gpt_4o_output.jsonl"),
    model_type="openai",
    trial_flg=True,
    model_name_judge=model_name_judge,
    )

# gpt 4 shot

out_dir = PROJDIR / "data/3_processed/dataset_2310/downstream/baseline/gpt_4o_1shot"
out_dir.mkdir(parents=True, exist_ok=True)
make_eval_batch(
    out_dir=str(out_dir),
    out_ext="eval_1shot_gpt_4o_",
    input_filename=str(out_dir / "gen_audres_1shot_gpt_4o_output.jsonl"),
    model_type="openai",
    trial_flg=True,
    model_name_judge=model_name_judge,
    )
# %% gpt 4o mini base

out_dir = PROJDIR / "data/3_processed/dataset_2310/downstream/baseline/gpt_4o_mini"
out_dir.mkdir(parents=True, exist_ok=True)
make_eval_batch(
    out_dir=str(out_dir),
    out_ext="eval_plane_gpt_4o_mini_",
    input_filename=str(out_dir / "aud_res_gpt_4o_mini_output.jsonl"),
    model_type="openai",
    trial_flg=True,
    model_name_judge=model_name_judge,
    )

# gpt 4o mini 1 shot

out_dir = PROJDIR / "data/3_processed/dataset_2310/downstream/baseline/gpt_4o_mini_shot"
out_dir.mkdir(parents=True, exist_ok=True)
make_eval_batch(
    out_dir=str(out_dir),
    out_ext="eval_1shot_gpt_4o_mini_",
    input_filename=str(out_dir / "batch_gen_audres_1shot_gpt_4o_mini_output.jsonl"),
    model_type="openai",
    trial_flg=True,
    model_name_judge=model_name_judge,
    )

# %% llama 3.1 plane

out_dir = PROJDIR / "data/3_processed/dataset_2310/downstream/baseline/llama_3.1_8b"
out_dir.mkdir(parents=True, exist_ok=True)
make_eval_batch(
    out_dir=str(out_dir),
    out_ext="eval_plane_llama_3.1_8b_",
    input_filename=str(out_dir / "gen_audres_llama_3.1_8b_output.csv"),
    model_type="llama",
    trial_flg=True,
    model_name_judge=model_name_judge,
    )

# llama 3.1 plane 1 shot

out_dir = PROJDIR / "data/3_processed/dataset_2310/downstream/baseline/llama_3.1_8b_1shot"
out_dir.mkdir(parents=True, exist_ok=True)
make_eval_batch(
    out_dir=str(out_dir),
    out_ext="eval_1shot_llama_3.1_8b_",
    input_filename=str(out_dir / "gen_audres_1shot_llama_3.1_8b_output.csv"),
    model_type="llama",
    trial_flg=True,
    model_name_judge=model_name_judge,
    )

# %% llama 3.1 sft merge

out_dir = PROJDIR / "data/3_processed/dataset_2310/downstream/3_processed/llama31_sft_merge"
out_dir.mkdir(parents=True, exist_ok=True)
make_eval_batch(
    out_dir=str(out_dir),
    out_ext="eval_llama_3.1_8b_",
    input_filename=str(out_dir / "eval_sft15k_jp_addon_instruct.csv"),
    model_type="llama",
    trial_flg=True,
    model_name_judge=model_name_judge,
    )
# llama 3.1 sft merge 1shot
out_dir = PROJDIR / "data/3_processed/dataset_2310/downstream/3_processed/llama31_sft_merge_1shot"
out_dir.mkdir(parents=True, exist_ok=True)
make_eval_batch(
    out_dir=str(out_dir),
    out_ext="eval_llama_3.1_8b_",
    input_filename=str(out_dir / "eval_sft15k_jp_addon_instruct_1shot.csv"),
    model_type="llama",
    trial_flg=True,
    model_name_judge=model_name_judge,
    )
# %% llama 3.1 sft instruct
out_dir = PROJDIR / "data/3_processed/dataset_2310/downstream/3_processed/llama31_sft_instruct"
out_dir.mkdir(parents=True, exist_ok=True)
make_eval_batch(
    out_dir=str(out_dir),
    out_ext="eval_llama_3.1_8b_",
    input_filename=str(out_dir / "eval_sft_15k_jp_instruct.csv"),
    model_type="llama",
    trial_flg=True,
    model_name_judge=model_name_judge,
    )

# llama 3.1 sft instruct 1shot

out_dir = PROJDIR / "data/3_processed/dataset_2310/downstream/3_processed/llama31_sft_instruct_1shot"
out_dir.mkdir(parents=True, exist_ok=True)
make_eval_batch(
    out_dir=str(out_dir),
    out_ext="eval_llama_3.1_8b_",
    input_filename=str(out_dir / "eval_sft_15k_jp_instruct_1shot.csv"),
    model_type="llama",
    trial_flg=True,
    model_name_judge=model_name_judge,
    )

# %%

# gemma 2b plane
# gemma 2b plane 1shot


# gemma 2b sft merge
# gemma 2b sft merge 1shot

# gemma 2b sft instruct
# gemma 2b sft instruct 1shot

# %% swallow

out_dir = PROJDIR / "data/3_processed/dataset_2310/downstream/baseline/swallow_8b"
out_dir.mkdir(parents=True, exist_ok=True)
make_eval_batch(
    out_dir=str(out_dir),
    out_ext="eval_1shot_swallow_",
    input_filename=str(out_dir / "eval_base_jp_instruct.csv"),
    model_type="llama",
    trial_flg=True,
    model_name_judge=model_name_judge,
    )

# %% swallow 1shot

out_dir = PROJDIR / "data/3_processed/dataset_2310/downstream/baseline/swallow_8b_1shot"
out_dir.mkdir(parents=True, exist_ok=True)
make_eval_batch(
    out_dir=str(out_dir),
    out_ext="eval_1shot_swallow_",
    input_filename=str(out_dir / "eval_base_jp_instruct_1shot.csv"),
    model_type="llama",
    trial_flg=True,
    model_name_judge=model_name_judge,
    )






# %%
# swallow sft merge
out_dir = PROJDIR / "data/3_processed/dataset_2310/downstream/3_processed/sft_merge_qlora"
out_dir.mkdir(parents=True, exist_ok=True)
make_eval_batch(
    out_dir=str(out_dir),
    out_ext="eval_sftqlora_merge_",
    input_filename=str(out_dir / "eval_sft7500_jp_addon_instruct.csv"),
    model_type="llama",
    trial_flg=True,
    model_name_judge=model_name_judge,
    )

# %% swallow sft merge 1shot
out_dir = PROJDIR / "data/3_processed/dataset_2310/downstream/3_processed/sft_merge_qlora_1shot"
out_dir.mkdir(parents=True, exist_ok=True)
make_eval_batch(
    out_dir=str(out_dir),
    out_ext="eval_sftqlora_merge_",
    input_filename=str(out_dir / "eval_sft15k_jp_addon_instruct_1shot.csv"),
    model_type="llama",
    trial_flg=True,
    model_name_judge=model_name_judge,
    )


# %%
# swallow sft instruct
out_dir = PROJDIR / "data/3_processed/dataset_2310/downstream/3_processed/sft_instract"
out_dir.mkdir(parents=True, exist_ok=True)
make_eval_batch(
    out_dir=str(out_dir),
    out_ext="eval_sftinst_merge_",
    input_filename=str(out_dir / "eval_sft_15k_jp_instruct.csv"),
    model_type="llama",
    trial_flg=True,
    model_name_judge=model_name_judge,
    )

# %%

# swallow sft instruct 1shot
out_dir = PROJDIR / "data/3_processed/dataset_2310/downstream/3_processed/sft_instruct_1shot"
out_dir.mkdir(parents=True, exist_ok=True)
make_eval_batch(
    out_dir=str(out_dir),
    out_ext="eval_sftinst_",
    input_filename=str(out_dir / "eval_sft_15k_jp_instruct_1shot.csv"),
    model_type="llama",
    trial_flg=True,
    model_name_judge=model_name_judge,
    )

# %%
# raft
out_dir = PROJDIR / "data/3_processed/dataset_2310/downstream/3_processed/raft_instruct"
out_dir.mkdir(parents=True, exist_ok=True)
make_eval_batch(
    out_dir=str(out_dir),
    out_ext="eval_raft_",
    input_filename=str(out_dir / "eval_laft_jp_instruct_1shot.csv"),
    model_type="llama",
    trial_flg=True,
    model_name_judge=model_name_judge,
    )

# %% moe
def prep_moe_concat_output(out_dir):
    
    input_files=out_dir.glob("inference_for_eval1127_c*.csv")
    out_all_df=pd.DataFrame()
    for file in input_files:
        print(file)
        out_df=pd.read_csv(file)
        out_all_df=pd.concat([out_all_df,out_df])
    print(out_all_df.shape)

    out_dir2 = PROJDIR / "data/3_processed/dataset_2310/downstream/3_processed/sft_instract"
    filename=str(out_dir2 / "eval_sft_15k_jp_instruct.csv")
    general_inf_df=pd.read_csv(filename)
    
    out_all_df=pd.merge(
        general_inf_df['index_num'],
        out_all_df,
        left_on='index_num',
        right_on='index_num',
        how='left')

    out_all_df.output=out_all_df.output.fillna(general_inf_df.output)

    #out_dir = PROJDIR / "data/3_processed/dataset_2310/downstream/3_processed/moe_sft"
    out_all_df.to_csv(out_dir/"eval_moesft_15k_jp_instruct.csv",index=False)

out_dir = PROJDIR / "data/3_processed/dataset_2310/downstream/3_processed/moe_sft/sft_and_topicsft"
#prep_moe_concat_output(out_dir)

out_dir = PROJDIR / "data/3_processed/dataset_2310/downstream/3_processed/moe_sft_continuous"
prep_moe_concat_output(out_dir)

# %%

filename=PROJDIR / "data/3_processed/dataset_2310/downstream" / "2_intermediate/llm_proc" /"audit_res_markdown_eval.csv"
data_val_df=pd.read_csv(filename,index_col=None,dtype=str).set_index('index_num').head(100)

# %%
out_dir = PROJDIR / "data/3_processed/dataset_2310/downstream/3_processed/moe_sft_continuous"
make_eval_batch(
    out_dir=str(out_dir),
    out_ext="eval_moesft_cont_",
    input_filename=str(out_dir / "eval_moesft_15k_jp_instruct.csv"),
    model_type="llama",
    trial_flg=True,
    model_name_judge=model_name_judge,
    )





# %% #####################################################
# eval llm-as-a-judge
##########################################################
#"/eval_data/aud_res_llama_3.1_8b_add_ans_0.2_output.csv"
#"/eval_data/aud_res_llama_3.1_8b_add_ans_0.4_output.csv"
#"/eval_data/aud_res_llama_3.1_8b_add_ans_0.6_output.csv"
#"/eval_data/aud_res_llama_3.1_8b_add_ans_0.8_output.csv"
#"/eval_data/batch_gen_audres_eval_abs_gpt_4o_mini_output.jsonl"
#"/eval_data/batch_gen_audres_eval_absfar_gpt_4o_mini_output.jsonl"
#"/eval_data/batch_gen_audres_eval_gpt_4o_mini_add_ans_far0.2_output.jsonl"
#"/eval_data/batch_gen_audres_eval_gpt_4o_mini_add_ans_far0.8_output.jsonl"
#"/eval_data/batch_gen_audres_eval_gpt_4o_mini_add_ans0.2_output.jsonl"
#"/eval_data/batch_gen_audres_eval_gpt_4o_mini_add_ans0.8_output.jsonl"
# gpt 4o mini base



out_dir = PROJDIR / "data/3_processed/dataset_2310/downstream/eval_data/aud_res_llama_3.1_8b_add_ans_0.2"
out_dir.mkdir(parents=True, exist_ok=True)
make_eval_batch(
    out_dir=str(out_dir),
    out_ext="llama_3.1_8b_add_ans_0.2_",
    input_filename=str(out_dir / "aud_res_llama_3.1_8b_add_ans_0.2_output.csv"),
    model_type="llama",
    trial_flg=True,
    model_name_judge=model_name_judge,
    )

out_dir = PROJDIR / "data/3_processed/dataset_2310/downstream/eval_data/aud_res_llama_3.1_8b_add_ans_0.4"
out_dir.mkdir(parents=True, exist_ok=True)

make_eval_batch(
    out_dir=str(out_dir),
    out_ext="llama_3.1_8b_add_ans_0.4_",
    input_filename=str(out_dir / "aud_res_llama_3.1_8b_add_ans_0.4_output.csv"),
    model_type="llama",
    trial_flg=True,
    model_name_judge=model_name_judge,
    )


out_dir = PROJDIR / "data/3_processed/dataset_2310/downstream/eval_data/aud_res_llama_3.1_8b_add_ans_0.6"
out_dir.mkdir(parents=True, exist_ok=True)

make_eval_batch(
    out_dir=str(out_dir),
    out_ext="llama_3.1_8b_add_ans_0.6_",
    input_filename=str(out_dir / "aud_res_llama_3.1_8b_add_ans_0.6_output.csv"),
    model_type="llama",
    trial_flg=True,
    model_name_judge=model_name_judge,
    )

out_dir = PROJDIR / "data/3_processed/dataset_2310/downstream/eval_data/aud_res_llama_3.1_8b_add_ans_0.8"
out_dir.mkdir(parents=True, exist_ok=True)
make_eval_batch(
    out_dir=str(out_dir),
    out_ext="llama_3.1_8b_add_ans_0.8_",
    input_filename=str(out_dir / "aud_res_llama_3.1_8b_add_ans_0.8_output.csv"),
    model_type="llama",
    trial_flg=True,
    model_name_judge=model_name_judge,
    )

out_dir = PROJDIR / "data/3_processed/dataset_2310/downstream/eval_data/eval_abs_gpt_4o_mini"
out_dir.mkdir(parents=True, exist_ok=True)

make_eval_batch(
    out_dir=str(out_dir),
    out_ext="abs_gpt_4o_mini_",
    input_filename=str(out_dir / "batch_gen_audres_eval_abs_gpt_4o_mini_output.jsonl"),
    model_type="openai",
    trial_flg=True,
    model_name_judge=model_name_judge,
    )

# %%
out_dir = PROJDIR / "data/3_processed/dataset_2310/downstream/eval_data/eval_abs_desc_gpt_4o_mini"
out_dir.mkdir(parents=True, exist_ok=True)

make_eval_batch(
    out_dir=str(out_dir),
    out_ext="abs_desc_gpt_4o_mini_",
    input_filename=str(out_dir / "batch_gen_audres_eval_abs_gpt_4o_mini_output.jsonl"),
    model_type="openai",
    trial_flg=True,
    model_name_judge=model_name_judge,
    )

# %%

out_dir = PROJDIR / "data/3_processed/dataset_2310/downstream/eval_data/eval_absfar_gpt_4o_mini"
out_dir.mkdir(parents=True, exist_ok=True)

make_eval_batch(
    out_dir=str(out_dir),
    out_ext="absfar_gpt_4o_mini_",
    input_filename=str(out_dir / "batch_gen_audres_eval_absfar_gpt_4o_mini_output.jsonl"),
    model_type="openai",
    trial_flg=True,
    model_name_judge=model_name_judge,
    )

out_dir = PROJDIR / "data/3_processed/dataset_2310/downstream/eval_data/eval_gpt_4o_mini_add_ans_far0.2"
out_dir.mkdir(parents=True, exist_ok=True)
# %%

out_dir = PROJDIR / "data/3_processed/dataset_2310/downstream/eval_data/eval_absfar_desc_gpt_4o_mini"
out_dir.mkdir(parents=True, exist_ok=True)

make_eval_batch(
    out_dir=str(out_dir),
    out_ext="absfar_desc_gpt_4o_mini_",
    input_filename=str(out_dir / "batch_gen_audres_eval_absfar_gpt_4o_mini_output.jsonl"),
    model_type="openai",
    trial_flg=True,
    model_name_judge=model_name_judge,
    )


# %%
out_dir = PROJDIR / "data/3_processed/dataset_2310/downstream/eval_data/eval_gpt_4o_mini_add_ans_far0.2"
out_dir.mkdir(parents=True, exist_ok=True)

make_eval_batch(
    out_dir=str(out_dir),
    out_ext="gpt_4o_mini_add_ans_far0.2_",
    input_filename=str(out_dir / "batch_gen_audres_eval_gpt_4o_mini_add_ans_far0.2_output.jsonl"),
    model_type="openai",
    trial_flg=True,
    model_name_judge=model_name_judge,
    )

out_dir = PROJDIR / "data/3_processed/dataset_2310/downstream/eval_data/eval_gpt_4o_mini_add_ans_far0.8"
out_dir.mkdir(parents=True, exist_ok=True)

make_eval_batch(
    out_dir=str(out_dir),
    out_ext="gpt_4o_mini_add_ans_far0.8_",
    input_filename=str(out_dir / "batch_gen_audres_eval_gpt_4o_mini_add_ans_far0.8_output.jsonl"),
    model_type="openai",
    trial_flg=True,
    model_name_judge=model_name_judge,
    )

out_dir = PROJDIR / "data/3_processed/dataset_2310/downstream/eval_data/eval_gpt_4o_mini_add_ans0.2"
out_dir.mkdir(parents=True, exist_ok=True)

make_eval_batch(
    out_dir=str(out_dir),
    out_ext="gpt_4o_mini_add_ans0.2_",
    input_filename=str(out_dir / "batch_gen_audres_eval_gpt_4o_mini_add_ans0.2_output.jsonl"),
    model_type="openai",
    trial_flg=True,
    model_name_judge=model_name_judge,
    )


out_dir = PROJDIR / "data/3_processed/dataset_2310/downstream/eval_data/eval_gpt_4o_mini_add_ans0.8"
out_dir.mkdir(parents=True, exist_ok=True)
make_eval_batch(
    out_dir=str(out_dir),
    out_ext="gpt_4o_mini_add_ans0.8_",
    input_filename=str(out_dir / "batch_gen_audres_eval_gpt_4o_mini_add_ans0.8_output.jsonl"),
    model_type="openai",
    trial_flg=True,
    model_name_judge=model_name_judge,
    )

# %%
