

# %%
from libs.model_api import *
from libs.compose_prompt import *
from libs.utils import *

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


# %%
PROJPATH=r"/Users/noro/Documents/Projects/XBRL_common_space_projection/"
PROJDIR=Path(PROJPATH)

# load train data
filename=PROJDIR / "data/3_processed/dataset_2310/downstream" / "3_processed/sft_data" /"data_train_markdown_1012.csv"
dict_df=pd.read_csv(filename,index_col=None,dtype=str).set_index('index_num')

# %%
# %%
#from openai import OpenAI
load_dotenv(verbose=True)
dotenv_path = join(Path(dirname(__file__)).parents[1] / "env" / "k", '.env')
load_dotenv(dotenv_path)

openai_api_obj=openai_api()
# %%

# %% plane llama 3.1

def default_prompt(sr:pd.Series):
    default_system_prompt="監査担当者であるあなたは、次の監査上の検討事項を与えられました。これに対応する監査上の対応事項を日本語文章で具体的に立案してください。"
    description_text=sr.description
    sys_prompt=default_system_prompt
    usr_prompt=description_text
    return sys_prompt,usr_prompt

def make_batch(dict_df,out_filename,model_name='gpt_4o_mini',prompt_gen_func=default_prompt):
    default_system_prompt="監査担当者であるあなたは、次の監査上の検討事項を与えられました。これに対応する監査上の対応事項を日本語文章で具体的に立案してください。"
    batch_inf_file_generator_obj=batch_inf_file_generator(
                model_name=model_name,
                )

    for index_num in tqdm(dict_df.index):
        id_text=dict_df.loc[index_num,'id']

        sys_prompt,usr_prompt=prompt_gen_func(dict_df.loc[index_num,:])
        #sys_prompt=default_system_prompt
        #usr_prompt=description_text
        itr_index_str=str(id_text)
        
        batch_inf_file_generator_obj.insert_inf_list_prompt(sys_prompt,usr_prompt,itr_index_str,max_tokens=1024, model_name=model_name)

    #out_filename=PROJDIR / "data/3_processed/dataset_2310/downstream" / "baseline" /("batch_gen_audres_"+model_name+".jsonl")
    batch_inf_file_generator_obj.export_list(out_filename)
    batch_inf_file_generator_obj.print_sample()
    
#model_name="llama_3.1_8b"
#out_filename=PROJDIR / "data/3_processed/dataset_2310/downstream" / "baseline" /("batch_gen_audres_"+model_name+".jsonl")
#make_batch(dict_df,out_filename,model_name=model_name,prompt_gen_func=default_prompt)



# %% 1-shot from training data
import csv
from transformers import AutoTokenizer, AutoModel
import torch
from torch import Tensor
import torch.nn.functional as F
from tqdm import tqdm


# %%

import csv
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from torch import Tensor
from scipy.spatial.distance import cdist

from pathlib import Path
import pandas as pd

# %%
PROJPATH=r"/Users/noro/Documents/Projects/XBRL_common_space_projection/"
PROJDIR=Path(PROJPATH)

class get_train_data_vector():
    def __init__(self):
        # all data
        filename=PROJDIR / "data/3_processed/dataset_2310/downstream" / "all_data_mapping" /"dict_all_df_2407_v1012_vect_df.pkl"
        vectors_df=pd.read_pickle(filename)

        # meta data included in cls_labels data
        filename=PROJDIR / "data/3_processed/dataset_2310/downstream/all_data_mapping" / "data_all17k_pivot_2407_v1012_with_cls.csv"
        data_all_pivot=pd.read_csv(filename)
        vectors_df.index=data_all_pivot.id
        self.vectors_df=vectors_df

        # training sample index
        data_train=pd.read_csv(PROJDIR / "data/3_processed/dataset_2310/downstream/3_processed/sft_data" / "data_train_1012.csv")
        train_id=set(data_train.id)
        self.data_train=data_train.set_index('id')
        # filter
        self.vectors_df_train=vectors_df.query("index in @train_id")

    def get_train_data_vector(self):
        return self.vectors_df_train
    
    def get_all_data_vector(self):
        return self.vectors_df

    def get_vecter(self,c_id):
        return self.vectors_df.loc[c_id,:]
    
    def get_nearest_id(self,vector,k=1):
        similarities = []
        for index, row in self.vectors_df_train.iterrows():
            similarity = cosine_similarity(vector, row)
            similarities.append((index, similarity))
        top_matches = sorted(similarities, key=lambda x: x[1], reverse=True)[:k]
        return top_matches

    def get_nearest_id_without_company(self,vector,c_id,k=1):
        edinet_code=GET_VECTOR_OBJ.data_train.loc[c_id,:].edinetCode
        #edinet_code='E01724'
        id_set=GET_VECTOR_OBJ.data_train.query("edinetCode != @edinet_code and (token_size_llama3_8b_audit_res+token_size_llama3_8b_description)<768").index

        similarities = []
        for index, row in self.vectors_df_train.query("index in @id_set").iterrows():
            similarity = cosine_similarity(vector, row)
            similarities.append((index, similarity))
        top_matches = sorted(similarities, key=lambda x: x[1], reverse=True)[:k]
        return top_matches

    def get_farthest_id(self,vector_list:list,k:int=1):
        similarities = []
        for index, row in self.vectors_df_train.iterrows():
            sim_list = []
            for vector in vector_list:
                sim_list.append(cosine_similarity(vector, row))
            # calc max
            similarities.append((index, max(sim_list)))
        top_matches = sorted(similarities, key=lambda x: x[1], reverse=False)[:k]
        return top_matches

    def train_instance(self,c_id):
        return self.data_train.loc[c_id,'description_latest'], self.data_train.loc[c_id,'audit_res_latest']
    
# %%


def cosine_similarity(v1, v2):
    return 1 - cdist([v1], [v2], 'cosine')[0][0]

# %% eval validation sample 項目反応理論?
import random
random.seed(42)
# %%
def few_shot_prompt(sr:pd.Series):
    instruction_1="監査担当者であるあなたは、監査上の検討事項を与えられたら、対応する監査上の対応事項を立案します。"
    instruction_2="例えば、次の検討事項が与えられました。\n\n#### 検討事項\n"
    instruction_3="\n\nこれに対応する監査上の対応事項は次のように立案されます。\n\n#### 監査上の対応事項\n"
    instruction_4="以上のように監査上の対応事項を日本語文章で具体的に立案してください。"
    c_id=sr.id
    description_text=sr.description
    tar_vec=GET_VECTOR_OBJ.get_vecter(c_id)
    k=sr.k_smp
    nearest_idx=GET_VECTOR_OBJ.get_nearest_id_without_company(tar_vec,c_id,k=k)
    #nearest_idx=random.sample(nearest_idx,1)
    shot_description_text, shot_audit_res = GET_VECTOR_OBJ.train_instance(nearest_idx[k-1][0])
    sys_prompt=instruction_1+instruction_2+shot_description_text+instruction_3+shot_audit_res+instruction_4
    usr_prompt="#### 検討事項\n"+description_text+"\n\n#### 監査上の対応事項"

    #sys_prompt=default_system_prompt
    #usr_prompt=description_text
    return sys_prompt,usr_prompt

GET_VECTOR_OBJ=get_train_data_vector()

#dict_df['k']=[random.sample(list(range(1,20+1)),12) for i in range(len(dict_df))]
#dict_df=dict_df.reset_index()
#dict_df.index_name='index_num_reset'
#for k_smp in range(1,12):
#    dict_df['k_smp']=dict_df['k'].apply(lambda x:x[k_smp])
#    model_name="llama_3.1_8b"
#    out_filename=PROJDIR / "data/3_processed/dataset_2310/downstream" / "3_processed/raft_train" /("batch_train_raft_1shot_"+str(k_smp)+"_"+model_name+".jsonl")
#    make_batch(dict_df,out_filename,model_name=model_name,prompt_gen_func=few_shot_prompt)

#model_name="gpt_4o_mini"
#out_filename=PROJDIR / "data/3_processed/dataset_2310/downstream" / "baseline" /("batch_gen_audres_1shot_"+model_name+".jsonl")
#make_batch(dict_df,out_filename,model_name=model_name,prompt_gen_func=few_shot_prompt)
# %%
# %%

def few_shot_prompt_mix(sr:pd.Series):
    instruction_1 = "監査担当者であるあなたは、監査上の検討事項を与えられたら、対応する監査上の対応事項を立案します。"
    instruction_2 = "例えば、次の検討事項が与えられました。\n\n#### 検討事項\n"
    instruction_3 = "\n\nこれに対応する監査上の対応事項は次のように立案されます。\n\n#### 監査上の対応事項\n"
    instruction_4 = "以上のように監査上の対応事項を日本語文章で具体的に立案してください。"
    c_id = sr.id
    description_text = sr.description
    tar_vec = GET_VECTOR_OBJ.get_vecter(c_id)
    k = sr.k_smp
    near_flg = sr.near_flg
    if near_flg>0:
        idx = GET_VECTOR_OBJ.get_nearest_id_without_company(tar_vec,c_id,k=k)
    else:
        idx = GET_VECTOR_OBJ.get_farthest_id([tar_vec],k=k)
    shot_description_text, shot_audit_res = GET_VECTOR_OBJ.train_instance(idx[k-1][0])
    sys_prompt = instruction_1 + instruction_2 + shot_description_text + instruction_3 + shot_audit_res + instruction_4
    usr_prompt = "#### 検討事項\n" + description_text + "\n\n#### 監査上の対応事項"

    #sys_prompt=default_system_prompt
    #usr_prompt=description_text
    return sys_prompt,usr_prompt

GET_VECTOR_OBJ=get_train_data_vector()

dict_df['k']=[random.sample(list(range(1,20+1)),6) for i in range(len(dict_df))]

dict_df=dict_df.reset_index()
dict_df.index_name='index_num_reset'
for k_smp in range(1,6):
    print(k_smp)
    dict_df['k_smp']=dict_df['k'].apply(lambda x:x[k_smp])
    dict_df['near_flg']=[random.randint(0,1) for i in range(len(dict_df))]
    model_name="llama_3.1_8b"
    out_filename=PROJDIR / "data/3_processed/dataset_2310/downstream" / "3_processed/sft_data/raft_train" /("batch_train_raft_1shot_p05_"+str(k_smp)+"_"+model_name+".jsonl")
    make_batch(dict_df,out_filename,model_name=model_name,prompt_gen_func=few_shot_prompt_mix)

# %%
