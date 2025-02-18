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
        model_name_judge=model_name_judge,
        model_name_judge_rel="gpt_4o",
        trial_flg=False
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
        eval_output_obj.gen_eval_batch_rel(proposed_res_list,out_filename_without_ext=out_ext+"_rel")
        eval_output_obj.gen_eval_batch(proposed_res_list,out_filename_without_ext=out_ext)
    elif model_type == "llama":
        inf_df=pd.read_csv(input_filename)
        proposed_res_list=inf_df['output'].to_list()
        if trial_flg:
            proposed_res_list=proposed_res_list#[:50]
        eval_output_obj.gen_eval_batch_rel(proposed_res_list,out_filename_without_ext=out_ext+"_rel")
        eval_output_obj.gen_eval_batch(proposed_res_list,out_filename_without_ext=out_ext)

    #filename="/Users/noro/Documents/Projects/XBRL_common_space_projection/data/3_processed/dataset_2310/downstream/3_processed/eval_sft7500_jp_addon_instruct.csv"


def make_eval_batch_hal(out_dir,out_ext,input_filename,model_type,trial_flg,model_name_judge="gpt_4o_mini",model_name_extract="gpt_4o_mini"):
    eval_prompt_path=str(PROJDIR / "data/3_processed/dataset_2310/downstream/eval_prep" / "eval_prompt_hal.json")
    eval_output_obj=eval_output(
        prompt_path=eval_prompt_path,
        out_dir=out_dir,
        model_name_extract="gpt_4o_mini",
        model_name_judge=model_name_judge,
        model_name_judge_rel="gpt_4o",
        trial_flg=False
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
        #eval_output_obj.gen_eval_batch_rel(proposed_res_list,out_filename_without_ext=out_ext+"_rel")
        #eval_output_obj.gen_eval_batch(proposed_res_list,out_filename_without_ext=out_ext)
        eval_output_obj.gen_and_expt_eval_batch_hal(proposed_res_list,out_filename_without_ext=out_ext)
    elif model_type == "llama":
        inf_df=pd.read_csv(input_filename)
        proposed_res_list=inf_df['output'].to_list()
        if trial_flg:
            proposed_res_list=proposed_res_list#[:50]
        #eval_output_obj.gen_eval_batch_rel(proposed_res_list,out_filename_without_ext=out_ext+"_rel")
        #eval_output_obj.gen_eval_batch(proposed_res_list,out_filename_without_ext=out_ext)
        eval_output_obj.gen_and_expt_eval_batch_hal(proposed_res_list,out_filename_without_ext=out_ext)
# %% script
#model_name_judge = "gpt_4o"
model_name_judge = "gpt_4_turbo"
# %%

calc_dict = {
    'llama31_sft_instruct_inv1shot':'instruct_llama31_far_smp_shot_0129.csv',
    'llama31_sft_merge_inv1shot':'merge_llama31_far_smp_shot_0130.csv',
    'raft_llama31_inv1shot':'raft_llama31_smp_shot_0130.csv',
    'raft_qwen_inv1shot':'raft_qwen_far_smp_shot_0129.csv',
    'raft_swallow_inv1shot':'raft_swallow_far_smp_shot_0129.csv',
    'sft_qwen_instruct_inv1shot':'instruct_qwen_far_smp_shot_0128.csv',
    'sft_qwen_merge_instruct_inv1shot':'merge_qwen_far_smp_shot_0128.csv',
    'swallow_sft_instruct_inv1shot':'instruct_swallow_far_smp_shot_0129.csv',
    'swallow_sft_merge_inv1shot':'merge_swallow_far_smp_shot_0129.csv'
    }

for key, val in calc_dict.items():
    print(key)
    out_dir = PROJDIR / f"data/3_processed/dataset_2310/downstream/3_processed/{key}"
    out_dir.mkdir(parents=True, exist_ok=True)
    make_eval_batch(
        out_dir=str(out_dir),
        out_ext=f"eval_{key}_",
        input_filename=str(out_dir / val),
        model_type="llama",
        trial_flg=False,
        model_name_judge=model_name_judge,
        )

# %% 20250131 qwen2_invknn4とllama31 nearfar_clsを追加実験


key = 'batch_gen_audres_many_shot_kNN4'
val = 'batch_gen_audres_many_shot_kNN4.csv'
out_dir = PROJDIR / f"data/3_processed/dataset_2310/downstream/few_shot_strategy_qwen2/{key}"
out_dir.mkdir(parents=True, exist_ok=True)
make_eval_batch(
    out_dir=str(out_dir),
    out_ext=f"eval_{key}_",
    input_filename=str(out_dir / val),
    model_type="llama",
    trial_flg=False,
    model_name_judge=model_name_judge,
    )

key = 'batch_gen_audres_many_shot_nearfar2_cls'
val = 'batch_gen_audres_many_shot_nearfar2_cls.csv'
out_dir = PROJDIR / f"data/3_processed/dataset_2310/downstream/few_shot_strategy/{key}"
out_dir.mkdir(parents=True, exist_ok=True)
make_eval_batch(
    out_dir=str(out_dir),
    out_ext=f"eval_{key}_",
    input_filename=str(out_dir / val),
    model_type="llama",
    trial_flg=False,
    model_name_judge=model_name_judge,
    )

# %% 20250202 qwen knn4を追加実験

key = 'batch_gen_audres_many_shot_kNN4'
val = 'batch_gen_audres_many_shot_kNN4.csv'
out_dir = PROJDIR / f"data/3_processed/dataset_2310/downstream/few_shot_strategy_qwen2/{key}"
out_dir.mkdir(parents=True, exist_ok=True)
make_eval_batch(
    out_dir=str(out_dir),
    out_ext=f"eval_{key}_",
    input_filename=str(out_dir / val),
    model_type="llama",
    trial_flg=False,
    model_name_judge=model_name_judge,
    )

# %% 20250204 qwen raft p05


key = 'raft_qwen_p05'
val = 'raft_qwen_p05.csv'
out_dir = PROJDIR / f"data/3_processed/dataset_2310/downstream/3_processed/raft_qwen_p05"
out_dir.mkdir(parents=True, exist_ok=True)
make_eval_batch(
    out_dir=str(out_dir),
    out_ext=f"eval_{key}_",
    input_filename=str(out_dir / val),
    model_type="llama",
    trial_flg=False,
    model_name_judge=model_name_judge,
    )
# %%

key = 'raft_swallow_p05'
val = 'raft_swallow_p05.csv'
out_dir = PROJDIR / f"data/3_processed/dataset_2310/downstream/3_processed/raft_swallow_p05"
out_dir.mkdir(parents=True, exist_ok=True)
make_eval_batch(
    out_dir=str(out_dir),
    out_ext=f"eval_{key}_",
    input_filename=str(out_dir / val),
    model_type="llama",
    trial_flg=False,
    model_name_judge=model_name_judge,
    )


key = 'raft_llama31_p05'
val = 'raft_llama31_p05.csv'
out_dir = PROJDIR / f"data/3_processed/dataset_2310/downstream/3_processed/raft_llama31_p05"
out_dir.mkdir(parents=True, exist_ok=True)
make_eval_batch(
    out_dir=str(out_dir),
    out_ext=f"eval_{key}_",
    input_filename=str(out_dir / val),
    model_type="llama",
    trial_flg=False,
    model_name_judge=model_name_judge,
    )
# %% hal 0215

calc_dict = {
    'llama31_sft_instruct_inv1shot':'instruct_llama31_far_smp_shot_0129.csv',
    'llama31_sft_merge_inv1shot':'merge_llama31_far_smp_shot_0130.csv',
    'raft_llama31_inv1shot':'raft_llama31_smp_shot_0130.csv',
    'raft_qwen_inv1shot':'raft_qwen_far_smp_shot_0129.csv',
    'raft_swallow_inv1shot':'raft_swallow_far_smp_shot_0129.csv',
    'sft_qwen_instruct_inv1shot':'instruct_qwen_far_smp_shot_0128.csv',
    'sft_qwen_merge_instruct_inv1shot':'merge_qwen_far_smp_shot_0128.csv',
    'swallow_sft_instruct_inv1shot':'instruct_swallow_far_smp_shot_0129.csv',
    'swallow_sft_merge_inv1shot':'merge_swallow_far_smp_shot_0129.csv'
    }

for key, val in calc_dict.items():
    print(key)
    out_dir = PROJDIR / f"data/3_processed/dataset_2310/downstream/3_processed/{key}"
    out_dir.mkdir(parents=True, exist_ok=True)
    make_eval_batch_hal(
        out_dir=str(out_dir),
        out_ext=f"eval_{key}_",
        input_filename=str(out_dir / val),
        model_type="llama",
        trial_flg=False,
        model_name_judge=model_name_judge,
        )


key = 'batch_gen_audres_many_shot_kNN4'
val = 'batch_gen_audres_many_shot_kNN4.csv'
out_dir = PROJDIR / f"data/3_processed/dataset_2310/downstream/few_shot_strategy_qwen2/{key}"
out_dir.mkdir(parents=True, exist_ok=True)
make_eval_batch_hal(
    out_dir=str(out_dir),
    out_ext=f"eval_{key}_",
    input_filename=str(out_dir / val),
    model_type="llama",
    trial_flg=False,
    model_name_judge=model_name_judge,
    )


key = 'batch_gen_audres_many_shot_kNN4'
val = 'batch_gen_audres_many_shot_kNN4.csv'
out_dir = PROJDIR / f"data/3_processed/dataset_2310/downstream/few_shot_strategy_qwen2/{key}"
out_dir.mkdir(parents=True, exist_ok=True)
make_eval_batch_hal(
    out_dir=str(out_dir),
    out_ext=f"eval_{key}_",
    input_filename=str(out_dir / val),
    model_type="llama",
    trial_flg=False,
    model_name_judge=model_name_judge,
    )


key = 'raft_qwen_p05'
val = 'raft_qwen_p05.csv'
out_dir = PROJDIR / f"data/3_processed/dataset_2310/downstream/3_processed/raft_qwen_p05"
out_dir.mkdir(parents=True, exist_ok=True)
make_eval_batch_hal(
    out_dir=str(out_dir),
    out_ext=f"eval_{key}_",
    input_filename=str(out_dir / val),
    model_type="llama",
    trial_flg=False,
    model_name_judge=model_name_judge,
    )
key = 'raft_swallow_p05'
val = 'raft_swallow_p05.csv'
out_dir = PROJDIR / f"data/3_processed/dataset_2310/downstream/3_processed/raft_swallow_p05"
out_dir.mkdir(parents=True, exist_ok=True)
make_eval_batch_hal(
    out_dir=str(out_dir),
    out_ext=f"eval_{key}_",
    input_filename=str(out_dir / val),
    model_type="llama",
    trial_flg=False,
    model_name_judge=model_name_judge,
    )


key = 'raft_llama31_p05'
val = 'raft_llama31_p05.csv'
out_dir = PROJDIR / f"data/3_processed/dataset_2310/downstream/3_processed/raft_llama31_p05"
out_dir.mkdir(parents=True, exist_ok=True)
make_eval_batch_hal(
    out_dir=str(out_dir),
    out_ext=f"eval_{key}_",
    input_filename=str(out_dir / val),
    model_type="llama",
    trial_flg=False,
    model_name_judge=model_name_judge,
    )








# %%
out_dir = PROJDIR / "data/3_processed/dataset_2310/downstream/baseline/gpt_4"
out_dir.mkdir(parents=True, exist_ok=True)
make_eval_batch(
    out_dir=str(out_dir),
    out_ext="eval_plane_gpt_4_",
    input_filename=str(out_dir / "aud_res_gpt_4_output.jsonl"),
    model_type="openai",
    trial_flg=False,
    model_name_judge=model_name_judge,
    )

# gpt 4 shot
out_dir = PROJDIR / "data/3_processed/dataset_2310/downstream/baseline/gpt_4_1shot"
out_dir.mkdir(parents=True, exist_ok=True)
make_eval_batch(
    out_dir=str(out_dir),
    out_ext="eval_1shot_gpt_4_",
    input_filename=str(out_dir / "gen_audres_1shot_gpt_4_output.jsonl"),
    model_type="openai",
    trial_flg=False,
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
    trial_flg=False,
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
    trial_flg=False,
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
    trial_flg=False,
    model_name_judge=model_name_judge,
    )

# gpt 4o mini 1 shot

out_dir = PROJDIR / "data/3_processed/dataset_2310/downstream/baseline/gpt_4o_mini_shot"
out_dir.mkdir(parents=True, exist_ok=True)
make_eval_batch(
    out_dir=str(out_dir),
    out_ext="eval_1shot_gpt_4o_mini_",
    input_filename=str(out_dir / "gen_audres_1shot_gpt_4o_mini_output.jsonl"),
    model_type="openai",
    trial_flg=False,
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
    trial_flg=False,
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
    trial_flg=False,
    model_name_judge=model_name_judge,
    )

# %% 20250126 llama 3.1 sft merge
out_dir = PROJDIR / "data/3_processed/dataset_2310/downstream/3_processed/llama31_sft_merge"
out_dir.mkdir(parents=True, exist_ok=True)
make_eval_batch(
    out_dir=str(out_dir),
    out_ext="eval_llama_3.1_8b_",
    input_filename=str(out_dir / "eval_sft15k_jp_addon_instruct.csv"),
    model_type="llama",
    trial_flg=False,
    model_name_judge=model_name_judge,
    )



# %% llama 3.1 sft merge 1shot
out_dir = PROJDIR / "data/3_processed/dataset_2310/downstream/3_processed/llama31_sft_merge_1shot"
out_dir.mkdir(parents=True, exist_ok=True)
make_eval_batch(
    out_dir=str(out_dir),
    out_ext="eval_llama_3.1_8b_",
    input_filename=str(out_dir / "eval_sft15k_jp_addon_instruct_1shot.csv"),
    model_type="llama",
    trial_flg=False,
    model_name_judge=model_name_judge,
    )
# %% llama 3.1 sft instruct
# will run 11/28
out_dir = PROJDIR / "data/3_processed/dataset_2310/downstream/3_processed/llama31_sft_instruct"
out_dir.mkdir(parents=True, exist_ok=True)
make_eval_batch(
    out_dir=str(out_dir),
    out_ext="eval_sft_0shot_llama_3.1_8b_",
    input_filename=str(out_dir / "eval_sft_15k_jp_instruct.csv"),
    model_type="llama",
    trial_flg=False,
    model_name_judge=model_name_judge,
    )

# %% llama 3.1 sft instruct 1shot

out_dir = PROJDIR / "data/3_processed/dataset_2310/downstream/3_processed/llama31_sft_instruct_1shot"
out_dir.mkdir(parents=True, exist_ok=True)
make_eval_batch(
    out_dir=str(out_dir),
    out_ext="eval_sft_llama_3.1_8b_",
    input_filename=str(out_dir / "eval_sft_15k_jp_instruct_1shot.csv"),
    model_type="llama",
    trial_flg=False,
    model_name_judge=model_name_judge,
    )
# %% moe
def prep_moe_concat_output_llama31(out_dir):
    """merge"""
    
    input_files=out_dir.glob("inference_for_eval1225_c*.csv")
    out_all_df=pd.DataFrame()#inference_for_eval1128_1shot_c0
    for file in input_files:
        print(file)
        out_df=pd.read_csv(file)
        out_all_df=pd.concat([out_all_df,out_df])
    print(out_all_df.shape)

    out_dir2 = PROJDIR / "data/3_processed/dataset_2310/downstream/3_processed/llama31_sft_instruct"
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

out_dir = PROJDIR / "data/3_processed/dataset_2310/downstream/3_processed/moe_llama31_instruct"
prep_moe_concat_output_llama31(out_dir)


# %%

# %% swallow #########################################################
#
# swallow 8b
#
######################################################################

out_dir = PROJDIR / "data/3_processed/dataset_2310/downstream/baseline/swallow_8b"
out_dir.mkdir(parents=True, exist_ok=True)
make_eval_batch(
    out_dir=str(out_dir),
    out_ext="eval_plane_swallow_",
    input_filename=str(out_dir / "eval_base_jp_instruct.csv"),
    model_type="llama",
    trial_flg=False,
    model_name_judge=model_name_judge,
    )

# %% swallow 1shot

out_dir = PROJDIR / "data/3_processed/dataset_2310/downstream/baseline/swallow_8b_1shot"
out_dir.mkdir(parents=True, exist_ok=True)
make_eval_batch(
    out_dir=str(out_dir),
    out_ext="eval_plane_1shot_swallow_",
    input_filename=str(out_dir / "eval_base_jp_instruct_1shot.csv"),
    model_type="llama",
    trial_flg=False,
    model_name_judge=model_name_judge,
    )






# %% swallow sft merge
out_dir = PROJDIR / "data/3_processed/dataset_2310/downstream/3_processed/sft_swallow_merge"
out_dir.mkdir(parents=True, exist_ok=True)
make_eval_batch(
    out_dir=str(out_dir),
    out_ext="eval_sft_swallow_merge_",
    input_filename=str(out_dir / "eval_sft15k_jp_addon_instruct_1129.csv"),
    model_type="llama",
    trial_flg=False,
    model_name_judge=model_name_judge,
    )

# %% swallow sft merge 1shot
out_dir = PROJDIR / "data/3_processed/dataset_2310/downstream/3_processed/sft_swallow_merge_1shot"
out_dir.mkdir(parents=True, exist_ok=True)
make_eval_batch(
    out_dir=str(out_dir),
    out_ext="eval_sft_swallow_merge_",
    input_filename=str(out_dir / "eval_sft15k_jp_addon_instruct_1shot.csv"),
    model_type="llama",
    trial_flg=False,
    model_name_judge=model_name_judge,
    )


# %%
# swallow sft instruct
out_dir = PROJDIR / "data/3_processed/dataset_2310/downstream/3_processed/sft_instruct"
out_dir.mkdir(parents=True, exist_ok=True)
make_eval_batch(
    out_dir=str(out_dir),
    out_ext="eval_sftinst_",
    input_filename=str(out_dir / "eval_sft_15k_jp_instruct.csv"),
    model_type="llama",
    trial_flg=False,
    model_name_judge=model_name_judge,
    )

# %%

# swallow sft instruct 1shot
out_dir = PROJDIR / "data/3_processed/dataset_2310/downstream/3_processed/sft_instruct_1shot"
out_dir.mkdir(parents=True, exist_ok=True)
make_eval_batch(
    out_dir=str(out_dir),
    out_ext="eval_sftinst_1shot_",
    input_filename=str(out_dir / "eval_sft_15k_jp_instruct_1shot.csv"),
    model_type="llama",
    trial_flg=False,
    model_name_judge=model_name_judge,
    )

# %%
# raft
#out_dir = PROJDIR / "data/3_processed/dataset_2310/downstream/3_processed/raft_instruct"
#out_dir.mkdir(parents=True, exist_ok=True)
#make_eval_batch(
#    out_dir=str(out_dir),
#    out_ext="eval_raft_",
#    input_filename=str(out_dir / "eval_laft_jp_instruct_1shot.csv"),
#    model_type="llama",
#    trial_flg=False,
#    model_name_judge=model_name_judge,
#    )

# %% moe
def prep_moe_concat_output(out_dir):
    
    input_files=out_dir.glob("inference_for_eval1128_1shot_c*.csv")
    out_all_df=pd.DataFrame()#inference_for_eval1128_1shot_c0
    for file in input_files:
        print(file)
        out_df=pd.read_csv(file)
        out_all_df=pd.concat([out_all_df,out_df])
    print(out_all_df.shape)

    out_dir2 = PROJDIR / "data/3_processed/dataset_2310/downstream/3_processed/sft_instruct_1shot"
    filename=str(out_dir2 / "eval_sft_15k_jp_instruct_1shot.csv")
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

out_dir = PROJDIR / "data/3_processed/dataset_2310/downstream/3_processed/moe_sft_continuous"
out_dir = PROJDIR / "data/3_processed/dataset_2310/downstream/3_processed/moe_sft_continuous_1shot"

# %%
def prep_moe_concat_output_swallow(out_dir):
    """merge"""
    
    input_files=out_dir.glob("inference_for_eval1225_c*.csv")
    out_all_df=pd.DataFrame()#inference_for_eval1128_1shot_c0
    for file in input_files:
        print(file)
        out_df=pd.read_csv(file)
        out_all_df=pd.concat([out_all_df,out_df])
    print(out_all_df.shape)

    out_dir2 = PROJDIR / "data/3_processed/dataset_2310/downstream/3_processed/sft_swallow_merge"
    filename=str(out_dir2 / "eval_sft15k_jp_addon_instruct_1129.csv")
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

out_dir = PROJDIR / "data/3_processed/dataset_2310/downstream/3_processed/moe_swallow_merge"
prep_moe_concat_output_swallow(out_dir)

# %% 20250119? moe general func

def prep_moe_concat(input_files:list,out_dir,general_inf_filename:str):
    
    #input_files=out_dir.glob("inference_for_eval1217_c*.csv")
    out_all_df=pd.DataFrame()
    for file in input_files:
        print(file)
        out_df=pd.read_csv(file)
        out_all_df=pd.concat([out_all_df,out_df])
    print(out_all_df.shape)

    general_inf_df=pd.read_csv(general_inf_filename)
    
    out_all_df=pd.merge(
        general_inf_df['index_num'],
        out_all_df,
        left_on='index_num',
        right_on='index_num',
        how='left')

    out_all_df.output=out_all_df.output.fillna(general_inf_df.output)
    out_all_df.to_csv(out_dir/"eval_moe_all.csv",index=False)

# %% 20250119? moe in qwen instruct

out_dir = PROJDIR / "data/3_processed/dataset_2310/downstream/3_processed/moe_qwen_instruct/infrst"
input_files=list(out_dir.glob("inference_for_eval0119_c*.csv"))
out_dir2 = PROJDIR / "data/3_processed/dataset_2310/downstream/3_processed/sft_qwen_instruct"
general_inf_filename=str(out_dir2 / "eval_sft_15k_jp_instruct.csv")
prep_moe_concat(input_files,out_dir,general_inf_filename)

out_dir = PROJDIR / "data/3_processed/dataset_2310/downstream/3_processed/moe_qwen_instruct"
make_eval_batch(
    out_dir=str(out_dir),
    out_ext="eval_moe_instruct_",
    input_filename=str(out_dir / "eval_moe_all.csv"),
    model_type="llama",
    trial_flg=False,
    model_name_judge=model_name_judge,
    )


# %% 20250126 moe in llama 31 pure

# instruct
out_dir = PROJDIR / "data/3_processed/dataset_2310/downstream/3_processed/moe_llama31_instruct/infrst"
input_files=list(out_dir.glob("inference_for_eval0125_c*.csv"))
out_dir2 = PROJDIR / "data/3_processed/dataset_2310/downstream/3_processed/llama31_sft_instruct"
general_inf_filename=str(out_dir2 / "eval_sft_15k_jp_instruct.csv")
prep_moe_concat(input_files,out_dir,general_inf_filename)

# merge
out_dir = PROJDIR / "data/3_processed/dataset_2310/downstream/3_processed/moe_llama31_merge/infrst"
input_files=list(out_dir.glob("inference_for_eval0126_c*.csv"))
out_dir2 = PROJDIR / "data/3_processed/dataset_2310/downstream/3_processed/llama31_sft_merge"
general_inf_filename=str(out_dir2 / "eval_sft15k_jp_addon_instruct.csv")
prep_moe_concat(input_files,out_dir,general_inf_filename)




# %%

out_dir = PROJDIR / "data/3_processed/dataset_2310/downstream/3_processed/moe_llama31_instruct"
make_eval_batch(
    out_dir=str(out_dir),
    out_ext="moe_llama31_instruct_",
    input_filename=str(out_dir / "eval_moe_all.csv"),
    model_type="llama",
    trial_flg=False,
    model_name_judge=model_name_judge,
    )

out_dir = PROJDIR / "data/3_processed/dataset_2310/downstream/3_processed/moe_llama31_merge"
make_eval_batch(
    out_dir=str(out_dir),
    out_ext="moe_llama31_merge_",
    input_filename=str(out_dir / "eval_moe_all.csv"),
    model_type="llama",
    trial_flg=False,
    model_name_judge=model_name_judge,
    )




# %%
def prep_moe_concat_output_qwen(out_dir):
    
    input_files=out_dir.glob("inference_for_eval1217_c*.csv")
    print(input_files)
    out_all_df=pd.DataFrame()#inference_for_eval1128_1shot_c0
    for file in input_files:
        print(file)
        out_df=pd.read_csv(file)
        out_all_df=pd.concat([out_all_df,out_df])
    print(out_all_df.shape)

    out_dir2 = PROJDIR / "data/3_processed/dataset_2310/downstream/3_processed/sft_qwen_merge_instruct"
    filename=str(out_dir2 / "eval_sft_qwen_addon_instruct_1216.csv")
    general_inf_df=pd.read_csv(filename)
    
    out_all_df=pd.merge(
        general_inf_df['index_num'],
        out_all_df,
        left_on='index_num',
        right_on='index_num',
        how='left')

    out_all_df.output=out_all_df.output.fillna(general_inf_df.output)

    #out_dir = PROJDIR / "data/3_processed/dataset_2310/downstream/3_processed/moe_sft"
    out_all_df.to_csv(out_dir/"eval_moe_all.csv",index=False)


out_dir = PROJDIR / "data/3_processed/dataset_2310/downstream/3_processed/moe_qwen_merge"
prep_moe_concat_output_qwen(out_dir)

def prep_moe_concat_output_qwen_1shot(out_dir):
    
    input_files=out_dir.glob("inference_for_eval1224_1shot_c*.csv")
    print(input_files)
    out_all_df=pd.DataFrame()#inference_for_eval1128_1shot_c0
    for file in input_files:
        print(file)
        out_df=pd.read_csv(file)
        out_all_df=pd.concat([out_all_df,out_df])
    print(out_all_df.shape)

    out_dir2 = PROJDIR / "data/3_processed/dataset_2310/downstream/3_processed/sft_qwen_merge_instruct_1shot/eval_cor_tokenizer"
    filename=str(out_dir2 / "eval_sft_qwen_addon_instruct_1shot_1224.csv")
    general_inf_df=pd.read_csv(filename)
    print(general_inf_df.shape)
    out_all_df=pd.merge(
        general_inf_df['index_num'],
        out_all_df,
        left_on='index_num',
        right_on='index_num',
        how='left')

    out_all_df.output=out_all_df.output.fillna(general_inf_df.output)

    #out_dir = PROJDIR / "data/3_processed/dataset_2310/downstream/3_processed/moe_sft"
    out_all_df.to_csv(out_dir/"eval_moe_all.csv",index=False)


out_dir = PROJDIR / "data/3_processed/dataset_2310/downstream/3_processed/moe_qwen_merge_1shot"
prep_moe_concat_output_qwen_1shot(out_dir)

# %%

filename=PROJDIR / "data/3_processed/dataset_2310/downstream" / "2_intermediate/llm_proc" /"audit_res_markdown_eval.csv"
data_val_df=pd.read_csv(filename,index_col=None,dtype=str).set_index('index_num').head(100)

# %%
out_dir = PROJDIR / "data/3_processed/dataset_2310/downstream/3_processed/moe_sft_continuous_1shot"
make_eval_batch(
    out_dir=str(out_dir),
    out_ext="eval_moesft_cont_",
    input_filename=str(out_dir / "eval_moesft_15k_jp_instruct.csv"),
    model_type="llama",
    trial_flg=False,
    model_name_judge=model_name_judge,
    )

# %%
out_dir = PROJDIR / "data/3_processed/dataset_2310/downstream/3_processed/moe_sft_continuous"
make_eval_batch(
    out_dir=str(out_dir),
    out_ext="eval_moesft_cont_",
    input_filename=str(out_dir / "eval_moesft_15k_jp_instruct.csv"),
    model_type="llama",
    trial_flg=False,
    model_name_judge=model_name_judge,
    )

# %% swallow merge moe
out_dir = PROJDIR / "data/3_processed/dataset_2310/downstream/3_processed/moe_swallow_merge"
make_eval_batch(
    out_dir=str(out_dir),
    out_ext="eval_moe_swallow_merge_",
    input_filename=str(out_dir / "eval_moesft_15k_jp_instruct.csv"),
    model_type="llama",
    trial_flg=False,
    model_name_judge=model_name_judge,
    )


# %% Qwen


out_dir = PROJDIR / "data/3_processed/dataset_2310/downstream/baseline/qwen_2_7b"
out_dir.mkdir(parents=True, exist_ok=True)
make_eval_batch(
    out_dir=str(out_dir),
    out_ext="eval_plane_qwen_",
    input_filename=str(out_dir / "eval_base_jp_instruct.csv"),
    model_type="llama",
    trial_flg=False,
    model_name_judge=model_name_judge,
    )

# %% qwen 1shot

out_dir = PROJDIR / "data/3_processed/dataset_2310/downstream/baseline/qwen_2_7b_1shot"
out_dir.mkdir(parents=True, exist_ok=True)

make_eval_batch(
    out_dir=str(out_dir),
    out_ext="eval_plane_1shot_qwen_",
    input_filename=str(out_dir / "eval_base_jp_instruct_1shot.csv"),
    model_type="llama",
    trial_flg=False,
    model_name_judge=model_name_judge,
    )
# %%
# qwen sft instruct
out_dir = PROJDIR / "data/3_processed/dataset_2310/downstream/3_processed/sft_qwen_instruct"
out_dir.mkdir(parents=True, exist_ok=True)

make_eval_batch(
    out_dir=str(out_dir),
    out_ext="eval_sft_qwen_inst_",
    input_filename=str(out_dir / "eval_sft_15k_jp_instruct.csv"),
    model_type="llama",
    trial_flg=False,
    model_name_judge=model_name_judge,
    )

# %% qwen sft instruct 1shot
out_dir = PROJDIR / "data/3_processed/dataset_2310/downstream/3_processed/sft_qwen_instruct_1shot"
out_dir.mkdir(parents=True, exist_ok=True)


make_eval_batch(
    out_dir=str(out_dir),
    out_ext="eval_sft_qwen_1shot_",
    input_filename=str(out_dir / "eval_sft_qwen_15k_instruct_1shot.csv"),
    model_type="llama",
    trial_flg=False,
    model_name_judge=model_name_judge,
    )

# %% qwen sft merge instruct
out_dir = PROJDIR / "data/3_processed/dataset_2310/downstream/3_processed/sft_qwen_merge_instruct"
out_dir.mkdir(parents=True, exist_ok=True)

make_eval_batch(
    out_dir=str(out_dir),
    out_ext="eval_sft_qwen_merge_",
    input_filename=str(out_dir / "eval_sft_qwen_addon_instruct_1216.csv"),
    model_type="llama",
    trial_flg=False,
    model_name_judge=model_name_judge,
    )

# %% qwen sft merge instruct 1shot
out_dir = PROJDIR / "data/3_processed/dataset_2310/downstream/3_processed/sft_qwen_merge_instruct_1shot"
out_dir.mkdir(parents=True, exist_ok=True)


make_eval_batch(
    out_dir=str(out_dir),
    out_ext="eval_sft_qwen_merge_1shot_",
    input_filename=str(out_dir / "eval_sft_qwen_addon_instruct_1shot.csv"),
    model_type="llama",
    trial_flg=False,
    model_name_judge=model_name_judge,
    )

# %% col tokenizer
out_dir = PROJDIR / "data/3_processed/dataset_2310/downstream/3_processed/sft_qwen_merge_instruct_1shot/eval_cor_tokenizer"
out_dir.mkdir(parents=True, exist_ok=True)


make_eval_batch(
    out_dir=str(out_dir),
    out_ext="eval_sft_qwen_merge_1shot_",
    input_filename=str(out_dir / "eval_sft_qwen_addon_instruct_1shot_1224.csv"),
    model_type="llama",
    trial_flg=False,
    model_name_judge=model_name_judge,
    )


# %% qwen moe merge

out_dir = PROJDIR / "data/3_processed/dataset_2310/downstream/3_processed/moe_qwen_merge"
make_eval_batch(
    out_dir=str(out_dir),
    out_ext="eval_qwen_moe_merge_",
    input_filename=str(out_dir / "eval_moe_all.csv"),
    model_type="llama",
    trial_flg=False,
    model_name_judge=model_name_judge,
    )

# %%
out_dir = PROJDIR / "data/3_processed/dataset_2310/downstream/3_processed/moe_qwen_merge_1shot"
make_eval_batch(
    out_dir=str(out_dir),
    out_ext="eval_qwen_moe_merge_1shot_",
    input_filename=str(out_dir / "eval_moe_all.csv"),
    model_type="llama",
    trial_flg=False,
    model_name_judge=model_name_judge,
    )


# %% raft
out_dir = PROJDIR / "data/3_processed/dataset_2310/downstream/3_processed/raft_instruct"
out_dir.mkdir(parents=True, exist_ok=True)
make_eval_batch(
    out_dir=str(out_dir),
    out_ext="eval_raft_",
    input_filename=str(out_dir / "eval_raft_jp_instruct_1shot.csv"),
    model_type="llama",
    trial_flg=False,
    model_name_judge=model_name_judge,
    )

# %%
# raft llama31
out_dir = PROJDIR / "data/3_processed/dataset_2310/downstream/3_processed/raft_llama31"
out_dir.mkdir(parents=True, exist_ok=True)
make_eval_batch(
    out_dir=str(out_dir),
    out_ext="eval_raft_llama31_",
    input_filename=str(out_dir / "eval_raft_llama31ins_1shot.csv"),
    model_type="llama",
    trial_flg=False,
    model_name_judge=model_name_judge,
    )

# %%
# raft qlora
out_dir = PROJDIR / "data/3_processed/dataset_2310/downstream/3_processed/raft_qwen"
out_dir.mkdir(parents=True, exist_ok=True)
make_eval_batch(
    out_dir=str(out_dir),
    out_ext="eval_raft_qwen_",
    input_filename=str(out_dir / "eval_raft_qwen_instruct_1shot.csv"),
    model_type="llama",
    trial_flg=False,
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
    trial_flg=False,
    model_name_judge=model_name_judge,
    )

out_dir = PROJDIR / "data/3_processed/dataset_2310/downstream/eval_data/aud_res_llama_3.1_8b_add_ans_0.4"
out_dir.mkdir(parents=True, exist_ok=True)

make_eval_batch(
    out_dir=str(out_dir),
    out_ext="llama_3.1_8b_add_ans_0.4_",
    input_filename=str(out_dir / "aud_res_llama_3.1_8b_add_ans_0.4_output.csv"),
    model_type="llama",
    trial_flg=False,
    model_name_judge=model_name_judge,
    )


out_dir = PROJDIR / "data/3_processed/dataset_2310/downstream/eval_data/aud_res_llama_3.1_8b_add_ans_0.6"
out_dir.mkdir(parents=True, exist_ok=True)

make_eval_batch(
    out_dir=str(out_dir),
    out_ext="llama_3.1_8b_add_ans_0.6_",
    input_filename=str(out_dir / "aud_res_llama_3.1_8b_add_ans_0.6_output.csv"),
    model_type="llama",
    trial_flg=False,
    model_name_judge=model_name_judge,
    )

out_dir = PROJDIR / "data/3_processed/dataset_2310/downstream/eval_data/aud_res_llama_3.1_8b_add_ans_0.8"
out_dir.mkdir(parents=True, exist_ok=True)
make_eval_batch(
    out_dir=str(out_dir),
    out_ext="llama_3.1_8b_add_ans_0.8_",
    input_filename=str(out_dir / "aud_res_llama_3.1_8b_add_ans_0.8_output.csv"),
    model_type="llama",
    trial_flg=False,
    model_name_judge=model_name_judge,
    )

out_dir = PROJDIR / "data/3_processed/dataset_2310/downstream/eval_data/eval_abs_gpt_4o_mini"
out_dir.mkdir(parents=True, exist_ok=True)

make_eval_batch(
    out_dir=str(out_dir),
    out_ext="abs_gpt_4o_mini_",
    input_filename=str(out_dir / "batch_gen_audres_eval_abs_gpt_4o_mini_output.jsonl"),
    model_type="openai",
    trial_flg=False,
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
    trial_flg=False,
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
    trial_flg=False,
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
    trial_flg=False,
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
    trial_flg=False,
    model_name_judge=model_name_judge,
    )

out_dir = PROJDIR / "data/3_processed/dataset_2310/downstream/eval_data/eval_gpt_4o_mini_add_ans_far0.8"
out_dir.mkdir(parents=True, exist_ok=True)

make_eval_batch(
    out_dir=str(out_dir),
    out_ext="gpt_4o_mini_add_ans_far0.8_",
    input_filename=str(out_dir / "batch_gen_audres_eval_gpt_4o_mini_add_ans_far0.8_output.jsonl"),
    model_type="openai",
    trial_flg=False,
    model_name_judge=model_name_judge,
    )

out_dir = PROJDIR / "data/3_processed/dataset_2310/downstream/eval_data/eval_gpt_4o_mini_add_ans0.2"
out_dir.mkdir(parents=True, exist_ok=True)

make_eval_batch(
    out_dir=str(out_dir),
    out_ext="gpt_4o_mini_add_ans0.2_",
    input_filename=str(out_dir / "batch_gen_audres_eval_gpt_4o_mini_add_ans0.2_output.jsonl"),
    model_type="openai",
    trial_flg=False,
    model_name_judge=model_name_judge,
    )


out_dir = PROJDIR / "data/3_processed/dataset_2310/downstream/eval_data/eval_gpt_4o_mini_add_ans0.8"
out_dir.mkdir(parents=True, exist_ok=True)
make_eval_batch(
    out_dir=str(out_dir),
    out_ext="gpt_4o_mini_add_ans0.8_",
    input_filename=str(out_dir / "batch_gen_audres_eval_gpt_4o_mini_add_ans0.8_output.jsonl"),
    model_type="openai",
    trial_flg=False,
    model_name_judge=model_name_judge,
    )

# %%
