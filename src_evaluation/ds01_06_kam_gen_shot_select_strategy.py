proc_text = """
few shot selection strategyの比較のために、各selection strategyのpromptを作成し、batchを生成します。
"""


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

# load validation data
filename=PROJDIR / "data/3_processed/dataset_2310/downstream" / "2_intermediate/llm_proc" /"audit_res_markdown_eval.csv"
dict_df=pd.read_csv(filename,index_col=None,dtype=str).set_index('index_num')
dict_df.shape
# %%
audit_res_text=dict_df.loc['1','output']
print(audit_res_text)
# %%
#from openai import OpenAI
load_dotenv(verbose=True)
dotenv_path = join(Path(dirname(__file__)).parents[1] / "env" / "k", '.env')
load_dotenv(dotenv_path)

openai_api_obj=openai_api()
# %%
def chk_api():
    itr_num=2
    description_text=dict_df.loc[itr_num,'description']
    default_system_prompt="監査担当者であるあなたは、次の監査上の検討事項を与えられました。これに対応する監査上の対応事項を日本語文章で具体的に立案してください。"

    output=openai_api_obj.request_rapper(
        system_prompt=default_system_prompt,
        usr_prompt=description_text,
        model_name='gpt_4o_mini',
        max_tokens=1024)

    output_text=output["output"]
    print(output['cost_dollar']*150)

# %% plane llama 3.1

def default_prompt(sr:pd.Series):
    default_system_prompt="監査担当者であるあなたは、次の監査上の検討事項を与えられました。これに対応する監査上の対応事項を日本語文章で具体的に立案してください。"
    description_text=sr.description
    sys_prompt=default_system_prompt
    usr_prompt=description_text
    return sys_prompt,usr_prompt

def make_batch(dict_df,out_filename,model_name='gpt_4o_mini',prompt_gen_func=default_prompt):
    """
    Generates a batch of prompts and saves them to a file.

    Args:
        dict_df (pd.DataFrame): DataFrame containing the data to generate prompts from.
        out_filename (str): The filename where the batch will be saved.
        model_name (str, optional): The name of the model to be used. Defaults to 'gpt_4o_mini'.
        prompt_gen_func (function, optional): Function to generate system and user prompts from the DataFrame row. Defaults to default_prompt.

    Returns:
        None
    """
    default_system_prompt="監査担当者であるあなたは、次の監査上の検討事項を与えられました。これに対応する監査上の対応事項を日本語文章で具体的に立案してください。"
    batch_inf_file_generator_obj=batch_inf_file_generator(
                model_name=model_name,
                )

    for index_num in tqdm(dict_df.index):
        sys_prompt,usr_prompt=prompt_gen_func(dict_df.loc[index_num,:])
        itr_index_str=str(index_num)
        batch_inf_file_generator_obj.insert_inf_list_prompt(sys_prompt,usr_prompt,itr_index_str,max_tokens=1024, model_name=model_name)

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


# %%
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

        # cluster data
        data_cls=pd.read_csv(PROJDIR / "data/3_processed/dataset_2310/downstream/all_data_mapping/data_all17k_pivot_2407_v1012_with_cls.csv")
        self.cls_label_df = data_cls.set_index('id')


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

    def get_cls_label(self,c_id:int)->str:
        return self.cls_label_df.loc[c_id,'cls_labels']

    def get_farthest_id_in_cls(self,vector_list:list,cls_label,k:int=1):
        id_set = self.cls_label_df.query("cls_labels==@cls_label").index
        similarities = []
        for index, row in self.vectors_df_train.query("index in @id_set").iterrows():
            sim_list = []
            for vector in vector_list:
                sim_list.append(cosine_similarity(vector, row))
            # calc max
            similarities.append((index, max(sim_list)))
        top_matches = sorted(similarities, key=lambda x: x[1], reverse=False)[:k]
        return top_matches

    
    def train_instance(self,c_id):
        return self.data_train.loc[c_id,'description_latest'], self.data_train.loc[c_id,'audit_res_latest']
    


class get_validation_data_vector():
    def __init__(self):
        # all data
        filename=PROJDIR / "data/3_processed/dataset_2310/downstream" / "all_data_mapping" /"dict_all_df_2407_v1012_vect_df.pkl"
        vectors_df=pd.read_pickle(filename)

        # meta data included in cls_labels data
        filename=PROJDIR / "data/3_processed/dataset_2310/downstream/all_data_mapping" / "data_all17k_pivot_2407_v1012_with_cls.csv"
        data_all_pivot=pd.read_csv(filename)
        vectors_df.index=data_all_pivot.id
        self.vectors_df=vectors_df

        # validation sample index
        data_validation=pd.read_csv(PROJDIR / "data/3_processed/dataset_2310/downstream/3_processed/sft_data" / "data_validation_1012.csv")
        validation_id=set(data_validation.id)
        self.data_validation=data_validation.set_index('id')
        # filter
        self.vectors_df_validation=vectors_df.query("index in @validation_id")

        # proc
        filename_openai_rst_proc=PROJDIR / "data/3_processed/dataset_2310/downstream" / "eval_prep" /"eval_extracted_process_from_ans_output.jsonl"
        response_list_df=pd.DataFrame(get_results_openai_batch(filename_openai_rst=filename_openai_rst_proc)).query("status!='Failed'")
        response_list_df=response_list_df.assign(
            index_num_df=response_list_df.index_num.str.replace("request-","")
            )
        response_list_df = pd.merge(response_list_df,data_validation,left_index=True,right_index=True,how='left')
        response_list_df = response_list_df.set_index('index_num_df').rename(
            columns={'status':'prep_status','output':'prep_output'}
            ).set_index('id')
        self.proc_df=response_list_df[['prep_output']]

    def get_validation_data_vector(self):
        return self.vectors_df_validation
    
    def get_all_data_vector(self):
        return self.vectors_df

    def get_vecter(self,c_id):
        return self.vectors_df.loc[c_id,:]
    
    def get_nearest_id(self,vector,k=1):
        similarities = []
        for index, row in self.vectors_df_validation.iterrows():
            similarity = cosine_similarity(vector, row)
            similarities.append((index, similarity))
        top_matches = sorted(similarities, key=lambda x: x[1], reverse=True)[:k]
        return top_matches
    
    def get_farthest_id(self,vector_list:list,k:int=1):
        similarities = []
        for index, row in self.vectors_df_validation.iterrows():
            similarity = -1*cosine_similarity(vector, row)
            similarities.append((index, similarity))
        top_matches = sorted(similarities, key=lambda x: x[1], reverse=True)[:k]
        return top_matches

    def get_farthest_id_in_cls(self,vector_list:list,k:int=1):
        similarities = []
        for index, row in self.vectors_df_validation.iterrows():
            similarity = -1*cosine_similarity(vector, row)
            similarities.append((index, similarity))
        top_matches = sorted(similarities, key=lambda x: x[1], reverse=True)[:k]
        return top_matches


    def validation_instance(self,c_id):
        return self.data_validation.loc[c_id,'description'], self.data_validation.loc[c_id,'audit_res']
    
    def proc_instance(self,c_id):
        return self.proc_df.loc[c_id,'prep_output']

def cosine_similarity(v1, v2):
    return 1 - cdist([v1], [v2], 'cosine')[0][0]

def chk_prompt():
    get_train_data_vector_obj=get_train_data_vector()
    index_num='0'
    c_id=dict_df.loc[index_num,'id']
    tar_vec=get_train_data_vector_obj.get_vecter(c_id)
    nearest_idx=get_train_data_vector_obj.get_nearest_id(tar_vec)
    shot_description_text, shot_audit_res = get_train_data_vector_obj.train_instance(nearest_idx[0][0])


# %% ################################################################
#
# eval validation sample 項目反応理論?
#
#####################################################################

GET_VECTOR_OBJ=get_train_data_vector()

nearest_score_list=[]
for index_num in tqdm(dict_df.head().index):
    c_id=dict_df.loc[index_num,'id']
    #'S100T3LA_FilingDateInstant_Row1Member'
    #description_text=sr.description
    tar_vec=GET_VECTOR_OBJ.get_vecter(c_id)
    nearest_idx=GET_VECTOR_OBJ.get_nearest_id(tar_vec,k=2)
    #shot_description_text, shot_audit_res = GET_VECTOR_OBJ.train_instance(nearest_idx[0][0])
    ant_t={
        "index_num":index_num,
        "c_id":c_id,
        "nearest_id":nearest_idx[0][0],
        "nearest_score":nearest_idx[0][1]
        }
    print(nearest_idx[0][0])
    nearest_score_list.append(ant_t)
nearest_score_list
#pd.DataFrame(nearest_score_list).to_csv(PROJDIR / "data/3_processed/dataset_2310/downstream" / "eval_prep" / "nearest_score.csv",index=None)
# %%
nearest_idx=GET_VECTOR_OBJ.get_nearest_id(tar_vec,k=2)
# %%
nearest_idx[1][0]

# %%

GET_VECTOR_OBJ=get_train_data_vector()


def few_shot_prompt_kNN_1(sr:pd.Series):
    instruction_1="監査担当者であるあなたは、監査上の検討事項を与えられたら、対応する監査上の対応事項を立案します。"
    instruction_2="例えば、次の検討事項が与えられました。\n\n#### 検討事項\n"
    instruction_3="\n\nこれに対応する監査上の対応事項は次のように立案されます。\n\n #### 監査上の対応事項\n"
    instruction_4="以上のように監査上の対応事項を日本語文章で具体的に立案してください。"
    c_id=sr.id
    description_text=sr.description
    tar_vec=GET_VECTOR_OBJ.get_vecter(c_id)
    nearest_idx=GET_VECTOR_OBJ.get_nearest_id(tar_vec,k=1)
    shot_description_text, shot_audit_res = GET_VECTOR_OBJ.train_instance(nearest_idx[0][0])
    sys_prompt=instruction_1+instruction_2+shot_description_text+instruction_3+shot_audit_res+instruction_4
    usr_prompt="#### 検討事項\n"+description_text+"\n\n#### 監査上の対応事項"

    return sys_prompt,usr_prompt


def few_shot_prompt_inv_kNN_1(sr:pd.Series):
    instruction_1="監査担当者であるあなたは、監査上の検討事項を与えられたら、対応する監査上の対応事項を立案します。"
    instruction_2="例えば、次の検討事項が与えられました。\n\n#### 検討事項\n"
    instruction_3="\n\nこれに対応する監査上の対応事項は次のように立案されます。\n\n #### 監査上の対応事項\n"
    instruction_4="以上のように監査上の対応事項を日本語文章で具体的に立案してください。"
    c_id=sr.id
    description_text=sr.description
    tar_vec=GET_VECTOR_OBJ.get_vecter(c_id)
    nearest_idx=GET_VECTOR_OBJ.get_farthest_id([tar_vec],k=1)
    shot_description_text, shot_audit_res = GET_VECTOR_OBJ.train_instance(nearest_idx[0][0])
    sys_prompt=instruction_1+instruction_2+shot_description_text+instruction_3+shot_audit_res+instruction_4
    usr_prompt="#### 検討事項\n"+description_text+"\n\n#### 監査上の対応事項"

    return sys_prompt,usr_prompt



def few_shot_prompt_rand_1(sr:pd.Series):
    instruction_1="監査担当者であるあなたは、監査上の検討事項を与えられたら、対応する監査上の対応事項を立案します。"
    instruction_2="例えば、次の検討事項が与えられました。\n\n#### 検討事項\n"
    instruction_3="\n\nこれに対応する監査上の対応事項は次のように立案されます。\n\n #### 監査上の対応事項\n"
    instruction_4="以上のように監査上の対応事項を日本語文章で具体的に立案してください。"
    c_id=sr.id
    description_text=sr.description
    #tar_vec=GET_VECTOR_OBJ.get_vecter(c_id)
    random_idx=np.random.choice(GET_VECTOR_OBJ.data_train.index,1)[0]
    shot_description_text, shot_audit_res = GET_VECTOR_OBJ.train_instance(random_idx)
    sys_prompt=instruction_1+instruction_2+shot_description_text+instruction_3+shot_audit_res+instruction_4
    usr_prompt="#### 検討事項\n"+description_text+"\n\n#### 監査上の対応事項"

    return sys_prompt,usr_prompt
#model_name="llama_3.1_8b"
#out_filename=PROJDIR / "data/3_processed/dataset_2310/downstream" / "baseline" /("batch_gen_audres_2shot_"+model_name+".jsonl")
#make_batch(dict_df,out_filename,model_name=model_name,prompt_gen_func=few_shot_prompt)


def few_shot_prompt_kNN_2(sr:pd.Series):
    instruction_1="監査担当者であるあなたは、監査上の検討事項を与えられたら、対応する監査上の対応事項を立案します。"
    instruction_2="例えば、次の検討事項が与えられました。\n\n##### 検討事項\n"
    instruction_3="\n\nこれに対応する監査上の対応事項は次のように立案されます。\n\n ##### 監査上の対応事項\n"
    instruction_4="以上のように監査上の対応事項を日本語文章で具体的に立案してください。"
    c_id=sr.id
    description_text=sr.description
    tar_vec=GET_VECTOR_OBJ.get_vecter(c_id)
    nearest_idx=GET_VECTOR_OBJ.get_nearest_id(tar_vec,k=2)
    shot_description_text, shot_audit_res = GET_VECTOR_OBJ.train_instance(nearest_idx[0][0])
    shot_description_text_2, shot_audit_res_2 = GET_VECTOR_OBJ.train_instance(nearest_idx[1][0])
    
    sys_prompt=(instruction_1
        +"\n\n#### 例1\n\n"+instruction_2
        +shot_description_text+instruction_3+shot_audit_res
        +"\n\n#### 例2\n\n"+instruction_2
        +shot_description_text_2+instruction_3+shot_audit_res_2
        +instruction_4)
    usr_prompt="#### 検討事項\n"+description_text+"\n\n#### 監査上の対応事項"

    return sys_prompt,usr_prompt


def few_shot_prompt_inv_kNN_2(sr:pd.Series):
    instruction_1="監査担当者であるあなたは、監査上の検討事項を与えられたら、対応する監査上の対応事項を立案します。"
    instruction_2="例えば、次の検討事項が与えられました。\n\n##### 検討事項\n"
    instruction_3="\n\nこれに対応する監査上の対応事項は次のように立案されます。\n\n ##### 監査上の対応事項\n"
    instruction_4="以上のように監査上の対応事項を日本語文章で具体的に立案してください。"
    c_id=sr.id
    description_text=sr.description
    tar_vec=GET_VECTOR_OBJ.get_vecter(c_id)
    farthest_idx=GET_VECTOR_OBJ.get_farthest_id([tar_vec],k=1)
    smp_1_vec=GET_VECTOR_OBJ.get_vecter(farthest_idx[0][0])
    farthest_idx_2=GET_VECTOR_OBJ.get_farthest_id([tar_vec,smp_1_vec],k=1)
    
    shot_description_text, shot_audit_res = GET_VECTOR_OBJ.train_instance(farthest_idx[0][0])
    shot_description_text_2, shot_audit_res_2 = GET_VECTOR_OBJ.train_instance(farthest_idx_2[0][0])
    
    sys_prompt=(instruction_1
        +"\n\n#### 例1\n\n"+instruction_2
        +shot_description_text+instruction_3+shot_audit_res
        +"\n\n#### 例2\n\n"+instruction_2
        +shot_description_text_2+instruction_3+shot_audit_res_2
        +instruction_4)
    usr_prompt="#### 検討事項\n"+description_text+"\n\n#### 監査上の対応事項"

    return sys_prompt,usr_prompt


def few_shot_prompt_rand_2(sr:pd.Series):
    instruction_1="監査担当者であるあなたは、監査上の検討事項を与えられたら、対応する監査上の対応事項を立案します。"
    instruction_2="例えば、次の検討事項が与えられました。\n\n##### 検討事項\n"
    instruction_3="\n\nこれに対応する監査上の対応事項は次のように立案されます。\n\n ##### 監査上の対応事項\n"
    instruction_4="以上のように監査上の対応事項を日本語文章で具体的に立案してください。"
    c_id=sr.id
    description_text=sr.description
    tar_vec=GET_VECTOR_OBJ.get_vecter(c_id)
    #nearest_idx=GET_VECTOR_OBJ.get_farthest_id(tar_vec,k=2)
    random_idx=np.random.choice(GET_VECTOR_OBJ.data_train.index,2)
    shot_description_text, shot_audit_res = GET_VECTOR_OBJ.train_instance(random_idx[0])
    shot_description_text_2, shot_audit_res_2 = GET_VECTOR_OBJ.train_instance(random_idx[1])

    sys_prompt=(instruction_1
        +"\n\n#### 例1\n\n"+instruction_2
        +shot_description_text+instruction_3+shot_audit_res
        +"\n\n#### 例2\n\n"+instruction_2
        +shot_description_text_2+instruction_3+shot_audit_res_2
        +instruction_4)
    usr_prompt="#### 検討事項\n"+description_text+"\n\n#### 監査上の対応事項"

    return sys_prompt,usr_prompt


def few_shot_prompt_near_and_far_2(sr:pd.Series):
    instruction_1="監査担当者であるあなたは、監査上の検討事項を与えられたら、対応する監査上の対応事項を立案します。"
    instruction_2="例えば、次の検討事項が与えられました。\n\n##### 検討事項\n"
    instruction_3="\n\nこれに対応する監査上の対応事項は次のように立案されます。\n\n ##### 監査上の対応事項\n"
    instruction_4="以上のように監査上の対応事項を日本語文章で具体的に立案してください。"
    c_id=sr.id
    description_text=sr.description
    tar_vec=GET_VECTOR_OBJ.get_vecter(c_id)
    nearest_idx=GET_VECTOR_OBJ.get_nearest_id(tar_vec,k=1)
    nearest_idx_f=GET_VECTOR_OBJ.get_farthest_id([tar_vec],k=1)
    
    shot_description_text, shot_audit_res = GET_VECTOR_OBJ.train_instance(nearest_idx[0][0])
    shot_description_text_2, shot_audit_res_2 = GET_VECTOR_OBJ.train_instance(nearest_idx_f[0][0])
    
    sys_prompt=(instruction_1
        +"\n\n#### 例1\n\n"+instruction_2
        +shot_description_text+instruction_3+shot_audit_res
        +"\n\n#### 例2\n\n"+instruction_2
        +shot_description_text_2+instruction_3+shot_audit_res_2
        +instruction_4)
    usr_prompt="#### 検討事項\n"+description_text+"\n\n#### 監査上の対応事項"

    return sys_prompt,usr_prompt


def few_shot_prompt_kNN_5(sr:pd.Series):
    instruction_1="監査担当者であるあなたは、監査上の検討事項を与えられたら、対応する監査上の対応事項を立案します。"
    instruction_2="例えば、次の検討事項が与えられました。\n\n##### 検討事項\n"
    instruction_3="\n\nこれに対応する監査上の対応事項は次のように立案されます。\n\n ##### 監査上の対応事項\n"
    instruction_4="以上のように監査上の対応事項を日本語文章で具体的に立案してください。"
    c_id=sr.id
    description_text=sr.description
    tar_vec=GET_VECTOR_OBJ.get_vecter(c_id)
    nearest_idx=GET_VECTOR_OBJ.get_nearest_id(tar_vec,k=5)
    shot_description_text, shot_audit_res = GET_VECTOR_OBJ.train_instance(nearest_idx[0][0])
    shot_description_text_2, shot_audit_res_2 = GET_VECTOR_OBJ.train_instance(nearest_idx[1][0])
    shot_description_text_3, shot_audit_res_3 = GET_VECTOR_OBJ.train_instance(nearest_idx[2][0])
    shot_description_text_4, shot_audit_res_4 = GET_VECTOR_OBJ.train_instance(nearest_idx[3][0])
    shot_description_text_5, shot_audit_res_5 = GET_VECTOR_OBJ.train_instance(nearest_idx[4][0])
    
    sys_prompt=(instruction_1
        +"\n\n#### 例1\n\n"+instruction_2
        +shot_description_text+instruction_3+shot_audit_res
        +"\n\n#### 例2\n\n"+instruction_2
        +shot_description_text_2+instruction_3+shot_audit_res_2
        +"\n\n#### 例3\n\n"+instruction_2
        +shot_description_text_3+instruction_3+shot_audit_res_3
        +"\n\n#### 例4\n\n"+instruction_2
        +shot_description_text_4+instruction_3+shot_audit_res_4
        +"\n\n#### 例5\n\n"+instruction_2
        +shot_description_text_5+instruction_3+shot_audit_res_5
        +instruction_4)
    usr_prompt="#### 検討事項\n"+description_text+"\n\n#### 監査上の対応事項"

    return sys_prompt,usr_prompt



def few_shot_prompt_inv_kNN_5(sr:pd.Series):
    instruction_1="監査担当者であるあなたは、監査上の検討事項を与えられたら、対応する監査上の対応事項を立案します。"
    instruction_2="例えば、次の検討事項が与えられました。\n\n##### 検討事項\n"
    instruction_3="\n\nこれに対応する監査上の対応事項は次のように立案されます。\n\n ##### 監査上の対応事項\n"
    instruction_4="以上のように監査上の対応事項を日本語文章で具体的に立案してください。"
    c_id=sr.id
    description_text=sr.description
    tar_vec=GET_VECTOR_OBJ.get_vecter(c_id)
    farthest_idx=GET_VECTOR_OBJ.get_farthest_id([tar_vec],k=1)
    smp_1_vec=GET_VECTOR_OBJ.get_vecter(farthest_idx[0][0])
    farthest_idx_2=GET_VECTOR_OBJ.get_farthest_id([tar_vec,smp_1_vec],k=1)
    smp_2_vec=GET_VECTOR_OBJ.get_vecter(farthest_idx_2[0][0])
    farthest_idx_3=GET_VECTOR_OBJ.get_farthest_id([tar_vec,smp_1_vec,smp_2_vec],k=1)
    smp_3_vec=GET_VECTOR_OBJ.get_vecter(farthest_idx_3[0][0])
    farthest_idx_4=GET_VECTOR_OBJ.get_farthest_id([tar_vec,smp_1_vec,smp_2_vec,smp_3_vec],k=1)
    smp_4_vec=GET_VECTOR_OBJ.get_vecter(farthest_idx_4[0][0])
    farthest_idx_5=GET_VECTOR_OBJ.get_farthest_id([tar_vec,smp_1_vec,smp_2_vec,smp_3_vec,smp_4_vec],k=1)

    shot_description_text, shot_audit_res = GET_VECTOR_OBJ.train_instance(farthest_idx[0][0])
    shot_description_text_2, shot_audit_res_2 = GET_VECTOR_OBJ.train_instance(farthest_idx_2[0][0])
    shot_description_text_3, shot_audit_res_3 = GET_VECTOR_OBJ.train_instance(farthest_idx_3[0][0])
    shot_description_text_4, shot_audit_res_4 = GET_VECTOR_OBJ.train_instance(farthest_idx_4[0][0])
    shot_description_text_5, shot_audit_res_5 = GET_VECTOR_OBJ.train_instance(farthest_idx_5[0][0])
    
    sys_prompt=(instruction_1
        +"\n\n#### 例1\n\n"+instruction_2
        +shot_description_text+instruction_3+shot_audit_res
        +"\n\n#### 例2\n\n"+instruction_2
        +shot_description_text_2+instruction_3+shot_audit_res_2
        +"\n\n#### 例3\n\n"+instruction_2
        +shot_description_text_3+instruction_3+shot_audit_res_3
        +"\n\n#### 例4\n\n"+instruction_2
        +shot_description_text_4+instruction_3+shot_audit_res_4
        +"\n\n#### 例5\n\n"+instruction_2
        +shot_description_text_5+instruction_3+shot_audit_res_5
        +instruction_4)
    usr_prompt="#### 検討事項\n"+description_text+"\n\n#### 監査上の対応事項"

    return sys_prompt,usr_prompt


def few_shot_prompt_near_and_far_4(sr:pd.Series):
    instruction_1="監査担当者であるあなたは、監査上の検討事項を与えられたら、対応する監査上の対応事項を立案します。"
    instruction_2="例えば、次の検討事項が与えられました。\n\n##### 検討事項\n"
    instruction_3="\n\nこれに対応する監査上の対応事項は次のように立案されます。\n\n ##### 監査上の対応事項\n"
    instruction_4="以上のように監査上の対応事項を日本語文章で具体的に立案してください。"
    c_id=sr.id
    description_text=sr.description
    tar_vec=GET_VECTOR_OBJ.get_vecter(c_id)
    nearest_idx=GET_VECTOR_OBJ.get_nearest_id(tar_vec,k=1)
    
    smp_1_vec=GET_VECTOR_OBJ.get_vecter(nearest_idx[0][0])
    farthest_idx_2=GET_VECTOR_OBJ.get_farthest_id([tar_vec,smp_1_vec],k=1)
    smp_2_vec=GET_VECTOR_OBJ.get_vecter(farthest_idx_2[0][0])
    farthest_idx_3=GET_VECTOR_OBJ.get_farthest_id([tar_vec,smp_1_vec,smp_2_vec],k=1)
    smp_3_vec=GET_VECTOR_OBJ.get_vecter(farthest_idx_3[0][0])
    farthest_idx_4=GET_VECTOR_OBJ.get_farthest_id([tar_vec,smp_1_vec,smp_2_vec,smp_3_vec],k=1)
    smp_4_vec=GET_VECTOR_OBJ.get_vecter(farthest_idx_4[0][0])
    farthest_idx_5=GET_VECTOR_OBJ.get_farthest_id([tar_vec,smp_1_vec,smp_2_vec,smp_3_vec,smp_4_vec],k=1)

    shot_description_text, shot_audit_res = GET_VECTOR_OBJ.train_instance(nearest_idx[0][0])
    shot_description_text_2, shot_audit_res_2 = GET_VECTOR_OBJ.train_instance(farthest_idx_2[0][0])
    shot_description_text_3, shot_audit_res_3 = GET_VECTOR_OBJ.train_instance(farthest_idx_3[0][0])
    shot_description_text_4, shot_audit_res_4 = GET_VECTOR_OBJ.train_instance(farthest_idx_4[0][0])
    shot_description_text_5, shot_audit_res_5 = GET_VECTOR_OBJ.train_instance(farthest_idx_5[0][0])
    
    sys_prompt=(instruction_1
        +"\n\n#### 例1\n\n"+instruction_2
        +shot_description_text+instruction_3+shot_audit_res
        +"\n\n#### 例2\n\n"+instruction_2
        +shot_description_text_2+instruction_3+shot_audit_res_2
        +"\n\n#### 例3\n\n"+instruction_2
        +shot_description_text_3+instruction_3+shot_audit_res_3
        +"\n\n#### 例4\n\n"+instruction_2
        +shot_description_text_4+instruction_3+shot_audit_res_4
        +"\n\n#### 例5\n\n"+instruction_2
        +shot_description_text_5+instruction_3+shot_audit_res_5
        +instruction_4)
    usr_prompt="#### 検討事項\n"+description_text+"\n\n#### 監査上の対応事項"

    return sys_prompt,usr_prompt


def few_shot_prompt_rand_5(sr:pd.Series):
    instruction_1="監査担当者であるあなたは、監査上の検討事項を与えられたら、対応する監査上の対応事項を立案します。"
    instruction_2="例えば、次の検討事項が与えられました。\n\n##### 検討事項\n"
    instruction_3="\n\nこれに対応する監査上の対応事項は次のように立案されます。\n\n ##### 監査上の対応事項\n"
    instruction_4="以上のように監査上の対応事項を日本語文章で具体的に立案してください。"
    c_id=sr.id
    description_text=sr.description
    tar_vec=GET_VECTOR_OBJ.get_vecter(c_id)
    #nearest_idx=GET_VECTOR_OBJ.get_farthest_id(tar_vec,k=5)
    random_idx=np.random.choice(GET_VECTOR_OBJ.data_train.index,5)
    shot_description_text, shot_audit_res = GET_VECTOR_OBJ.train_instance(random_idx[0])
    shot_description_text_2, shot_audit_res_2 = GET_VECTOR_OBJ.train_instance(random_idx[1])
    shot_description_text_3, shot_audit_res_3 = GET_VECTOR_OBJ.train_instance(random_idx[2])
    shot_description_text_4, shot_audit_res_4 = GET_VECTOR_OBJ.train_instance(random_idx[3])
    shot_description_text_5, shot_audit_res_5 = GET_VECTOR_OBJ.train_instance(random_idx[4])
    
    sys_prompt=(instruction_1
        +"\n\n#### 例1\n\n"+instruction_2
        +shot_description_text+instruction_3+shot_audit_res
        +"\n\n#### 例2\n\n"+instruction_2
        +shot_description_text_2+instruction_3+shot_audit_res_2
        +"\n\n#### 例3\n\n"+instruction_2
        +shot_description_text_3+instruction_3+shot_audit_res_3
        +"\n\n#### 例4\n\n"+instruction_2
        +shot_description_text_4+instruction_3+shot_audit_res_4
        +"\n\n#### 例5\n\n"+instruction_2
        +shot_description_text_5+instruction_3+shot_audit_res_5
        +instruction_4)
    usr_prompt="#### 検討事項\n"+description_text+"\n\n#### 監査上の対応事項"

    return sys_prompt,usr_prompt

# %%

model_name='llama_3.1_8b'
default_system_prompt="監査担当者であるあなたは、次の監査上の検討事項を与えられました。これに対応する監査上の対応事項を日本語文章で具体的に立案してください。"
batch_inf_file_generator_obj=batch_inf_file_generator(
            model_name=model_name,
            )
#for index_num in tqdm(dict_df.index):
#    sys_prompt,usr_prompt=few_shot_prompt_kNN_1(dict_df.loc[index_num,:])
#    itr_index_str="kNN1_"+str(index_num)
#    batch_inf_file_generator_obj.insert_inf_list_prompt(sys_prompt,usr_prompt,itr_index_str,max_tokens=1024, model_name=model_name)

for index_num in tqdm(dict_df.index):
    sys_prompt,usr_prompt=few_shot_prompt_inv_kNN_1(dict_df.loc[index_num,:])
    itr_index_str="invkNN1_"+str(index_num)
    batch_inf_file_generator_obj.insert_inf_list_prompt(sys_prompt,usr_prompt,itr_index_str,max_tokens=1024, model_name=model_name)

for index_num in tqdm(dict_df.index):
    sys_prompt,usr_prompt=few_shot_prompt_rand_1(dict_df.loc[index_num,:])
    itr_index_str="rand1_"+str(index_num)
    batch_inf_file_generator_obj.insert_inf_list_prompt(sys_prompt,usr_prompt,itr_index_str,max_tokens=1024, model_name=model_name)

for index_num in tqdm(dict_df.index):
    sys_prompt,usr_prompt=few_shot_prompt_kNN_2(dict_df.loc[index_num,:])
    itr_index_str="kNN2_"+str(index_num)
    batch_inf_file_generator_obj.insert_inf_list_prompt(sys_prompt,usr_prompt,itr_index_str,max_tokens=1024, model_name=model_name)

for index_num in tqdm(dict_df.index):
    sys_prompt,usr_prompt=few_shot_prompt_inv_kNN_2(dict_df.loc[index_num,:])
    itr_index_str="invkNN2_"+str(index_num)
    batch_inf_file_generator_obj.insert_inf_list_prompt(sys_prompt,usr_prompt,itr_index_str,max_tokens=1024, model_name=model_name)

for index_num in tqdm(dict_df.index):
    sys_prompt,usr_prompt=few_shot_prompt_rand_2(dict_df.loc[index_num,:])
    itr_index_str="rand2_"+str(index_num)
    batch_inf_file_generator_obj.insert_inf_list_prompt(sys_prompt,usr_prompt,itr_index_str,max_tokens=1024, model_name=model_name)

for index_num in tqdm(dict_df.index):
    sys_prompt,usr_prompt=few_shot_prompt_near_and_far_2(dict_df.loc[index_num,:])
    itr_index_str="nearfar2_"+str(index_num)
    batch_inf_file_generator_obj.insert_inf_list_prompt(sys_prompt,usr_prompt,itr_index_str,max_tokens=1024, model_name=model_name)

for index_num in tqdm(dict_df.index):
    sys_prompt,usr_prompt=few_shot_prompt_kNN_5(dict_df.loc[index_num,:])
    itr_index_str="kNN5_"+str(index_num)
    batch_inf_file_generator_obj.insert_inf_list_prompt(sys_prompt,usr_prompt,itr_index_str,max_tokens=1024, model_name=model_name)

for index_num in tqdm(dict_df.index):
    sys_prompt,usr_prompt=few_shot_prompt_inv_kNN_5(dict_df.loc[index_num,:])
    itr_index_str="invkNN5_"+str(index_num)
    batch_inf_file_generator_obj.insert_inf_list_prompt(sys_prompt,usr_prompt,itr_index_str,max_tokens=1024, model_name=model_name)

for index_num in tqdm(dict_df.index):
    sys_prompt,usr_prompt=few_shot_prompt_near_and_far_4(dict_df.loc[index_num,:])
    itr_index_str="nearfar4_"+str(index_num)
    batch_inf_file_generator_obj.insert_inf_list_prompt(sys_prompt,usr_prompt,itr_index_str,max_tokens=1024, model_name=model_name)

for index_num in tqdm(dict_df.index):
    sys_prompt,usr_prompt=few_shot_prompt_rand_5(dict_df.loc[index_num,:])
    itr_index_str="rand5_"+str(index_num)
    batch_inf_file_generator_obj.insert_inf_list_prompt(sys_prompt,usr_prompt,itr_index_str,max_tokens=1024, model_name=model_name)

out_filename=PROJDIR / "data/3_processed/dataset_2310/downstream" / "few_shot_strategy" /("batch_gen_audres_many_shot_"+model_name+".jsonl")
batch_inf_file_generator_obj.export_list(out_filename)
batch_inf_file_generator_obj.print_sample()
    
# %%
filename = "/Users/noro/Documents/Projects/XBRL_common_space_projection/data/3_processed/dataset_2310/downstream/few_shot_strategy/batch_gen_audres_many_shot_llama_3.1_8b.csv"
pred_df = pd.read_csv(filename)

# %%
strategy_list = pred_df.index_num.str.replace("request-","").str.split("_",expand=True).rename(columns={0:"strategy",1:"index_num"}).groupby("strategy").count().index.tolist()
# %%
strategy_list
# %%
for strategy in strategy_list:
    out_dir = PROJDIR / "data/3_processed/dataset_2310/downstream" / "few_shot_strategy" / f"batch_gen_audres_many_shot_{strategy}"
    out_dir.mkdir(parents=True,exist_ok=True)
    pred_df.query("index_num.str.contains(@strategy)").to_csv(out_dir / f"batch_gen_audres_many_shot_{strategy}.csv",index=None)
# %% make eval batch




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

# %% script
#model_name_judge = "gpt_4o"
model_name_judge = "gpt_4_turbo"


for strategy in strategy_list:
    out_dir = PROJDIR / "data/3_processed/dataset_2310/downstream/few_shot_strategy" / f"batch_gen_audres_many_shot_{strategy}"
    out_dir.mkdir(parents=True, exist_ok=True)
    make_eval_batch(
        out_dir=str(out_dir),
        out_ext=f"batch_gen_audres_many_shot_{strategy}"+"_",
        input_filename=str(out_dir / (f"batch_gen_audres_many_shot_{strategy}"+".csv")),
        model_type="llama",
        trial_flg=False,
        model_name_judge=model_name_judge,
        )


# %% take 2
GET_VECTOR_OBJ=get_train_data_vector()


def few_shot_prompt_kNN_3(sr:pd.Series):
    instruction_1="監査担当者であるあなたは、監査上の検討事項を与えられたら、対応する監査上の対応事項を立案します。"
    instruction_2="例えば、次の検討事項が与えられました。\n\n##### 検討事項\n"
    instruction_3="\n\nこれに対応する監査上の対応事項は次のように立案されます。\n\n ##### 監査上の対応事項\n"
    instruction_4="以上のように監査上の対応事項を日本語文章で具体的に立案してください。"
    c_id=sr.id
    description_text=sr.description
    tar_vec=GET_VECTOR_OBJ.get_vecter(c_id)
    nearest_idx=GET_VECTOR_OBJ.get_nearest_id(tar_vec,k=5)
    shot_description_text, shot_audit_res = GET_VECTOR_OBJ.train_instance(nearest_idx[0][0])
    shot_description_text_2, shot_audit_res_2 = GET_VECTOR_OBJ.train_instance(nearest_idx[1][0])
    shot_description_text_3, shot_audit_res_3 = GET_VECTOR_OBJ.train_instance(nearest_idx[2][0])
    #shot_description_text_4, shot_audit_res_4 = GET_VECTOR_OBJ.train_instance(nearest_idx[3][0])
    #shot_description_text_5, shot_audit_res_5 = GET_VECTOR_OBJ.train_instance(nearest_idx[4][0])
    
    sys_prompt=(instruction_1
        +"\n\n#### 例1\n\n"+instruction_2
        +shot_description_text+instruction_3+shot_audit_res
        +"\n\n#### 例2\n\n"+instruction_2
        +shot_description_text_2+instruction_3+shot_audit_res_2
        +"\n\n#### 例3\n\n"+instruction_2
        +shot_description_text_3+instruction_3+shot_audit_res_3
        #+"\n\n#### 例4\n\n"+instruction_2
        #+shot_description_text_4+instruction_3+shot_audit_res_4
        #+"\n\n#### 例5\n\n"+instruction_2
        #+shot_description_text_5+instruction_3+shot_audit_res_5
        +"\n\n"+instruction_4)
    usr_prompt="#### 検討事項\n"+description_text+"\n\n#### 監査上の対応事項"

    return sys_prompt,usr_prompt



def few_shot_prompt_inv_kNN_3(sr:pd.Series):
    instruction_1="監査担当者であるあなたは、監査上の検討事項を与えられたら、対応する監査上の対応事項を立案します。"
    instruction_2="例えば、次の検討事項が与えられました。\n\n##### 検討事項\n"
    instruction_3="\n\nこれに対応する監査上の対応事項は次のように立案されます。\n\n ##### 監査上の対応事項\n"
    instruction_4="以上のように監査上の対応事項を日本語文章で具体的に立案してください。"
    c_id=sr.id
    description_text=sr.description
    tar_vec=GET_VECTOR_OBJ.get_vecter(c_id)
    farthest_idx=GET_VECTOR_OBJ.get_farthest_id([tar_vec],k=1)
    smp_1_vec=GET_VECTOR_OBJ.get_vecter(farthest_idx[0][0])
    farthest_idx_2=GET_VECTOR_OBJ.get_farthest_id([tar_vec,smp_1_vec],k=1)
    smp_2_vec=GET_VECTOR_OBJ.get_vecter(farthest_idx_2[0][0])
    farthest_idx_3=GET_VECTOR_OBJ.get_farthest_id([tar_vec,smp_1_vec,smp_2_vec],k=1)
    smp_3_vec=GET_VECTOR_OBJ.get_vecter(farthest_idx_3[0][0])
    farthest_idx_4=GET_VECTOR_OBJ.get_farthest_id([tar_vec,smp_1_vec,smp_2_vec,smp_3_vec],k=1)
    smp_4_vec=GET_VECTOR_OBJ.get_vecter(farthest_idx_4[0][0])
    farthest_idx_5=GET_VECTOR_OBJ.get_farthest_id([tar_vec,smp_1_vec,smp_2_vec,smp_3_vec,smp_4_vec],k=1)

    shot_description_text, shot_audit_res = GET_VECTOR_OBJ.train_instance(farthest_idx[0][0])
    shot_description_text_2, shot_audit_res_2 = GET_VECTOR_OBJ.train_instance(farthest_idx_2[0][0])
    shot_description_text_3, shot_audit_res_3 = GET_VECTOR_OBJ.train_instance(farthest_idx_3[0][0])
    #shot_description_text_4, shot_audit_res_4 = GET_VECTOR_OBJ.train_instance(farthest_idx_4[0][0])
    #shot_description_text_5, shot_audit_res_5 = GET_VECTOR_OBJ.train_instance(farthest_idx_5[0][0])
    
    sys_prompt=(instruction_1
        +"\n\n#### 例1\n\n"+instruction_2
        +shot_description_text+instruction_3+shot_audit_res
        +"\n\n#### 例2\n\n"+instruction_2
        +shot_description_text_2+instruction_3+shot_audit_res_2
        +"\n\n#### 例3\n\n"+instruction_2
        +shot_description_text_3+instruction_3+shot_audit_res_3
        #+"\n\n#### 例4\n\n"+instruction_2
        #+shot_description_text_4+instruction_3+shot_audit_res_4
        #+"\n\n#### 例5\n\n"+instruction_2
        #+shot_description_text_5+instruction_3+shot_audit_res_5
        +"\n\n"+instruction_4)
    usr_prompt="#### 検討事項\n"+description_text+"\n\n#### 監査上の対応事項"

    return sys_prompt,usr_prompt


def few_shot_prompt_near_and_far_2(sr:pd.Series):
    instruction_1="監査担当者であるあなたは、監査上の検討事項を与えられたら、対応する監査上の対応事項を立案します。"
    instruction_2="例えば、次の検討事項が与えられました。\n\n##### 検討事項\n"
    instruction_3="\n\nこれに対応する監査上の対応事項は次のように立案されます。\n\n ##### 監査上の対応事項\n"
    instruction_4="以上のように監査上の対応事項を日本語文章で具体的に立案してください。"
    c_id=sr.id
    description_text=sr.description
    tar_vec=GET_VECTOR_OBJ.get_vecter(c_id)
    nearest_idx=GET_VECTOR_OBJ.get_nearest_id(tar_vec,k=1)
    
    smp_1_vec=GET_VECTOR_OBJ.get_vecter(nearest_idx[0][0])
    farthest_idx_2=GET_VECTOR_OBJ.get_farthest_id([tar_vec,smp_1_vec],k=1)
    smp_2_vec=GET_VECTOR_OBJ.get_vecter(farthest_idx_2[0][0])
    farthest_idx_3=GET_VECTOR_OBJ.get_farthest_id([tar_vec,smp_1_vec,smp_2_vec],k=1)
    #smp_3_vec=GET_VECTOR_OBJ.get_vecter(farthest_idx_3[0][0])
    #farthest_idx_4=GET_VECTOR_OBJ.get_farthest_id([tar_vec,smp_1_vec,smp_2_vec,smp_3_vec],k=1)
    #smp_4_vec=GET_VECTOR_OBJ.get_vecter(farthest_idx_4[0][0])
    #farthest_idx_5=GET_VECTOR_OBJ.get_farthest_id([tar_vec,smp_1_vec,smp_2_vec,smp_3_vec,smp_4_vec],k=1)

    shot_description_text, shot_audit_res = GET_VECTOR_OBJ.train_instance(nearest_idx[0][0])
    shot_description_text_2, shot_audit_res_2 = GET_VECTOR_OBJ.train_instance(farthest_idx_2[0][0])
    shot_description_text_3, shot_audit_res_3 = GET_VECTOR_OBJ.train_instance(farthest_idx_3[0][0])
    #shot_description_text_4, shot_audit_res_4 = GET_VECTOR_OBJ.train_instance(farthest_idx_4[0][0])
    #shot_description_text_5, shot_audit_res_5 = GET_VECTOR_OBJ.train_instance(farthest_idx_5[0][0])
    
    sys_prompt=(instruction_1
        +"\n\n#### 例1\n\n"+instruction_2
        +shot_description_text+instruction_3+shot_audit_res
        +"\n\n#### 例2\n\n"+instruction_2
        +shot_description_text_2+instruction_3+shot_audit_res_2
        +"\n\n#### 例3\n\n"+instruction_2
        +shot_description_text_3+instruction_3+shot_audit_res_3
        #+"\n\n#### 例4\n\n"+instruction_2
        #+shot_description_text_4+instruction_3+shot_audit_res_4
        #+"\n\n#### 例5\n\n"+instruction_2
        #+shot_description_text_5+instruction_3+shot_audit_res_5
        +"\n\n"+instruction_4)
    usr_prompt="#### 検討事項\n"+description_text+"\n\n#### 監査上の対応事項"

    return sys_prompt,usr_prompt


def few_shot_prompt_rand_3(sr:pd.Series):
    instruction_1="監査担当者であるあなたは、監査上の検討事項を与えられたら、対応する監査上の対応事項を立案します。"
    instruction_2="例えば、次の検討事項が与えられました。\n\n##### 検討事項\n"
    instruction_3="\n\nこれに対応する監査上の対応事項は次のように立案されます。\n\n ##### 監査上の対応事項\n"
    instruction_4="以上のように監査上の対応事項を日本語文章で具体的に立案してください。"
    c_id=sr.id
    description_text=sr.description
    tar_vec=GET_VECTOR_OBJ.get_vecter(c_id)
    #nearest_idx=GET_VECTOR_OBJ.get_farthest_id(tar_vec,k=5)
    random_idx=np.random.choice(GET_VECTOR_OBJ.data_train.index,5)
    shot_description_text, shot_audit_res = GET_VECTOR_OBJ.train_instance(random_idx[0])
    shot_description_text_2, shot_audit_res_2 = GET_VECTOR_OBJ.train_instance(random_idx[1])
    shot_description_text_3, shot_audit_res_3 = GET_VECTOR_OBJ.train_instance(random_idx[2])
    #shot_description_text_4, shot_audit_res_4 = GET_VECTOR_OBJ.train_instance(random_idx[3])
    #shot_description_text_5, shot_audit_res_5 = GET_VECTOR_OBJ.train_instance(random_idx[4])
    
    sys_prompt=(instruction_1
        +"\n\n#### 例1\n\n"+instruction_2
        +shot_description_text+instruction_3+shot_audit_res
        +"\n\n#### 例2\n\n"+instruction_2
        +shot_description_text_2+instruction_3+shot_audit_res_2
        +"\n\n#### 例3\n\n"+instruction_2
        +shot_description_text_3+instruction_3+shot_audit_res_3
        #+"\n\n#### 例4\n\n"+instruction_2
        #+shot_description_text_4+instruction_3+shot_audit_res_4
        #+"\n\n#### 例5\n\n"+instruction_2
        #+shot_description_text_5+instruction_3+shot_audit_res_5
        +"\n\n"+instruction_4)
    usr_prompt="#### 検討事項\n"+description_text+"\n\n#### 監査上の対応事項"

    return sys_prompt,usr_prompt




model_name='llama_3.1_8b'
default_system_prompt="監査担当者であるあなたは、次の監査上の検討事項を与えられました。これに対応する監査上の対応事項を日本語文章で具体的に立案してください。"
batch_inf_file_generator_obj=batch_inf_file_generator(
            model_name=model_name,
            )

for index_num in tqdm(dict_df.index):
    sys_prompt,usr_prompt=few_shot_prompt_kNN_3(dict_df.loc[index_num,:])
    itr_index_str="kNN3_"+str(index_num)
    batch_inf_file_generator_obj.insert_inf_list_prompt(sys_prompt,usr_prompt,itr_index_str,max_tokens=1024, model_name=model_name)
batch_inf_file_generator_obj.print_sample()


for index_num in tqdm(dict_df.index):
    sys_prompt,usr_prompt=few_shot_prompt_inv_kNN_3(dict_df.loc[index_num,:])
    itr_index_str="invkNN3_"+str(index_num)
    batch_inf_file_generator_obj.insert_inf_list_prompt(sys_prompt,usr_prompt,itr_index_str,max_tokens=1024, model_name=model_name)

batch_inf_file_generator_obj.print_sample()

for index_num in tqdm(dict_df.index):
    sys_prompt,usr_prompt=few_shot_prompt_near_and_far_2(dict_df.loc[index_num,:])
    itr_index_str="nearfar3_"+str(index_num)
    batch_inf_file_generator_obj.insert_inf_list_prompt(sys_prompt,usr_prompt,itr_index_str,max_tokens=1024, model_name=model_name)
batch_inf_file_generator_obj.print_sample()


for index_num in tqdm(dict_df.index):
    sys_prompt,usr_prompt=few_shot_prompt_rand_3(dict_df.loc[index_num,:])
    itr_index_str="rand3_"+str(index_num)
    batch_inf_file_generator_obj.insert_inf_list_prompt(sys_prompt,usr_prompt,itr_index_str,max_tokens=1024, model_name=model_name)

batch_inf_file_generator_obj.print_sample()

out_filename=PROJDIR / "data/3_processed/dataset_2310/downstream" / "few_shot_strategy" /("batch_gen_audres_many_shot2_"+model_name+".jsonl")
batch_inf_file_generator_obj.export_list(out_filename)

# %% step 2_2

filename = "/Users/noro/Documents/Projects/XBRL_common_space_projection/data/3_processed/dataset_2310/downstream/few_shot_strategy/batch_gen_audres_many_shot2_llama_3.1_8b.csv"
pred_df = pd.read_csv(filename)
strategy_list = pred_df.index_num.str.replace("request-","").str.split("_",expand=True).rename(columns={0:"strategy",1:"index_num"}).groupby("strategy").count().index.tolist()
# %%
strategy_list
# %% divide by strategy
for strategy in strategy_list:
    out_dir = PROJDIR / "data/3_processed/dataset_2310/downstream" / "few_shot_strategy" / f"batch_gen_audres_many_shot_{strategy}"
    out_dir.mkdir(parents=True,exist_ok=True)
    pred_df.query("index_num.str.contains(@strategy)").to_csv(out_dir / f"batch_gen_audres_many_shot_{strategy}.csv",index=None)
# %% make eval batch
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

# %% script
from libs.kam_evaluation import *

#model_name_judge = "gpt_4o"
model_name_judge = "gpt_4_turbo"

for strategy in strategy_list:
    out_dir = PROJDIR / "data/3_processed/dataset_2310/downstream/few_shot_strategy" / f"batch_gen_audres_many_shot_{strategy}"
    out_dir.mkdir(parents=True, exist_ok=True)
    make_eval_batch(
        out_dir=str(out_dir),
        out_ext=f"batch_gen_audres_many_shot_{strategy}"+"_",
        input_filename=str(out_dir / (f"batch_gen_audres_many_shot_{strategy}"+".csv")),
        model_type="llama",
        trial_flg=False,
        model_name_judge=model_name_judge,
        )

# %% retry openai errror
import json
model_name = "llama_3.1_8b"
filename=PROJDIR / "data/3_processed/dataset_2310/downstream" / "few_shot_strategy" /"batch_gen_audres_many_shot_nearfar3/batch_gen_audres_many_shot_nearfar3__4o_test.jsonl"
with open(filename) as f:
    data = f.readlines()
    data = [json.loads(line) for line in data]

data_f = [data_l for data_l in data if data_l["custom_id"] == 'request-llmcomc-154']
out_filename=PROJDIR / "data/3_processed/dataset_2310/downstream" / "few_shot_strategy" /"batch_gen_audres_many_shot_nearfar3/batch_gen_audres_many_shot_nearfar3__4o_test_2.jsonl"

with open(out_filename, 'w') as file:
    for obj in data_f:
        file.write(json.dumps(obj) + '\n')

# %% 20250128 make 1 and 2 inf batch
model_name = "llama_3.1_8b"
filename=PROJDIR / "data/3_processed/dataset_2310/downstream" / "few_shot_strategy" /("batch_gen_audres_many_shot_"+model_name+".jsonl")
with open(filename) as f:
    data_1 = f.readlines()
    data_1 = [json.loads(line) for line in data_1]

filename=PROJDIR / "data/3_processed/dataset_2310/downstream" / "few_shot_strategy" /("batch_gen_audres_many_shot2_"+model_name+".jsonl")

with open(filename) as f:
    data_2 = f.readlines()
    data_2 = [json.loads(line) for line in data_2]

data = data_1 + data_2


# %% ##########################################################################
#
# 20250128 make inv inf
#
###############################################################################
model_name = "llama_3.1_8b"
filename=PROJDIR / "data/3_processed/dataset_2310/downstream" / "few_shot_strategy" /("batch_gen_audres_many_shot_"+model_name+".jsonl")
with open(filename) as f:
    data_1 = f.readlines()
    data_1 = [json.loads(line) for line in data_1]

filename=PROJDIR / "data/3_processed/dataset_2310/downstream" / "few_shot_strategy" /("batch_gen_audres_many_shot2_"+model_name+".jsonl")

with open(filename) as f:
    data_2 = f.readlines()
    data_2 = [json.loads(line) for line in data_2]

data = data_1 + data_2
data_f = [
    data_l for data_l in data 
    if ('request-invkNN5' not in data_l["custom_id"])&('request-rand5' not in data_l["custom_id"])&('request-nearfar4' not in data_l["custom_id"])
    ]

# %%
# save all inf set
out_filename = PROJDIR / "data/3_processed/dataset_2310/downstream" / "few_shot_strategy" /("batch_gen_audres_many_shot_1and2_"+model_name+".jsonl")
with open(out_filename, 'w') as file:
    for obj in data_f:
        file.write(json.dumps(obj) + '\n')


# %% 4 make eval batch

from libs.kam_evaluation import *
# input
data_f_df = pd.DataFrame(data_f)
# output qwen
filename = "/Users/noro/Documents/Projects/XBRL_common_space_projection/data/3_processed/dataset_2310/downstream/few_shot_strategy_qwen2/inf_rag_0127.csv"
pred_df = pd.read_csv(filename)
pred_df = pd.merge(pred_df,data_f_df,left_on="index_num",right_index=True)

for strategy in strategy_list:
    out_dir = PROJDIR / "data/3_processed/dataset_2310/downstream" / "few_shot_strategy_qwen2" / f"batch_gen_audres_many_shot_{strategy}"
    out_dir.mkdir(parents=True,exist_ok=True)
    pred_df.query("custom_id.str.contains(@strategy)").to_csv(out_dir / f"batch_gen_audres_many_shot_{strategy}.csv",index=None)

model_name_judge = "gpt_4_turbo"
strategy_list = pred_df.custom_id.str.replace("request-","").str.split("_",expand=True).rename(columns={0:"strategy",1:"index_num"}).groupby("strategy").count().index.tolist()
for strategy in strategy_list:
    out_dir = PROJDIR / "data/3_processed/dataset_2310/downstream/few_shot_strategy_qwen2" / f"batch_gen_audres_many_shot_{strategy}"
    out_dir.mkdir(parents=True, exist_ok=True)
    make_eval_batch(
        out_dir=str(out_dir),
        out_ext=f"batch_gen_audres_many_shot_{strategy}"+"_",
        input_filename=str(out_dir / (f"batch_gen_audres_many_shot_{strategy}"+".csv")),
        model_type="llama",
        trial_flg=False,
        model_name_judge=model_name_judge,
        )

# %%
# output swallow
filename = "/Users/noro/Documents/Projects/XBRL_common_space_projection/data/3_processed/dataset_2310/downstream/few_shot_strategy_swallow/inf_rag_0125.csv"
pred_df = pd.read_csv(filename)
pred_df = pd.merge(pred_df,data_f_df,left_on="index_num",right_index=True)

for strategy in strategy_list:
    out_dir = PROJDIR / "data/3_processed/dataset_2310/downstream" / "few_shot_strategy_swallow" / f"batch_gen_audres_many_shot_{strategy}"
    out_dir.mkdir(parents=True,exist_ok=True)
    pred_df.query("custom_id.str.contains(@strategy)").to_csv(out_dir / f"batch_gen_audres_many_shot_{strategy}.csv",index=None)

model_name_judge = "gpt_4_turbo"
strategy_list = pred_df.custom_id.str.replace("request-","").str.split("_",expand=True).rename(columns={0:"strategy",1:"index_num"}).groupby("strategy").count().index.tolist()
for strategy in strategy_list:
    out_dir = PROJDIR / "data/3_processed/dataset_2310/downstream/few_shot_strategy_swallow" / f"batch_gen_audres_many_shot_{strategy}"
    out_dir.mkdir(parents=True, exist_ok=True)
    make_eval_batch(
        out_dir=str(out_dir),
        out_ext=f"batch_gen_audres_many_shot_{strategy}"+"_",
        input_filename=str(out_dir / (f"batch_gen_audres_many_shot_{strategy}"+".csv")),
        model_type="llama",
        trial_flg=False,
        model_name_judge=model_name_judge,
        )







# %% ##########################################################################
#
# 20250228 gen inv-kNN1 batch
#
###############################################################################
data_f = [
    data_l for data_l in data 
    if ('request-invkNN1' in data_l["custom_id"])
    ]

# save inv-knn1
out_filename = PROJDIR / "data/3_processed/dataset_2310/downstream" / "few_shot_strategy" /("batch_gen_audres_far_smp_shot_"+model_name+".jsonl")
with open(out_filename, 'w') as file:
    for obj in data_f:
        file.write(json.dumps(obj) + '\n')




# %% ############################################################################### 
#
# 20250129 add hallucination eval
#
####################################################################################

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

# %%

filename = "/Users/noro/Documents/Projects/XBRL_common_space_projection/data/3_processed/dataset_2310/downstream/few_shot_strategy/batch_gen_audres_many_shot_llama_3.1_8b.csv"
pred_df = pd.read_csv(filename)
strategy_list = pred_df.index_num.str.replace("request-","").str.split("_",expand=True).rename(columns={0:"strategy",1:"index_num"}).groupby("strategy").count().index.tolist()
strategy_list

filename = "/Users/noro/Documents/Projects/XBRL_common_space_projection/data/3_processed/dataset_2310/downstream/few_shot_strategy/batch_gen_audres_many_shot2_llama_3.1_8b.csv"
pred_df = pd.read_csv(filename)
strategy_list2 = pred_df.index_num.str.replace("request-","").str.split("_",expand=True).rename(columns={0:"strategy",1:"index_num"}).groupby("strategy").count().index.tolist()
# %%
strategy_list = strategy_list + strategy_list2
strategy_list
# %%




#from openai import OpenAI
load_dotenv(verbose=True)
dotenv_path = join(PROJDIR / "env" / "k", '.env')
load_dotenv(dotenv_path)
openai_api_obj=openai_api()

# %% script
from libs.kam_evaluation import *

#model_name_judge = "gpt_4o"
model_name_judge = "gpt_4_turbo"

for strategy in strategy_list:
    out_dir = PROJDIR / "data/3_processed/dataset_2310/downstream/few_shot_strategy" / f"batch_gen_audres_many_shot_{strategy}"
    out_dir.mkdir(parents=True, exist_ok=True)
    make_eval_batch_hal(
        out_dir=str(out_dir),
        out_ext=f"batch_gen_audres_many_shot_{strategy}"+"_",
        input_filename=str(out_dir / (f"batch_gen_audres_many_shot_{strategy}"+".csv")),
        model_type="llama",
        trial_flg=False,
        model_name_judge=model_name_judge,
        )


from libs.kam_evaluation import *
# input
data_f_df = pd.DataFrame(data_f)
# output qwen
filename = "/Users/noro/Documents/Projects/XBRL_common_space_projection/data/3_processed/dataset_2310/downstream/few_shot_strategy_qwen2/inf_rag_0127.csv"
pred_df = pd.read_csv(filename)
pred_df = pd.merge(pred_df,data_f_df,left_on="index_num",right_index=True)

for strategy in strategy_list:
    out_dir = PROJDIR / "data/3_processed/dataset_2310/downstream" / "few_shot_strategy_qwen2" / f"batch_gen_audres_many_shot_{strategy}"
    out_dir.mkdir(parents=True,exist_ok=True)
    pred_df.query("custom_id.str.contains(@strategy)").to_csv(out_dir / f"batch_gen_audres_many_shot_{strategy}.csv",index=None)

model_name_judge = "gpt_4_turbo"
strategy_list = pred_df.custom_id.str.replace("request-","").str.split("_",expand=True).rename(columns={0:"strategy",1:"index_num"}).groupby("strategy").count().index.tolist()
for strategy in strategy_list:
    out_dir = PROJDIR / "data/3_processed/dataset_2310/downstream/few_shot_strategy_qwen2" / f"batch_gen_audres_many_shot_{strategy}"
    out_dir.mkdir(parents=True, exist_ok=True)
    make_eval_batch_hal(
        out_dir=str(out_dir),
        out_ext=f"batch_gen_audres_many_shot_{strategy}"+"_",
        input_filename=str(out_dir / (f"batch_gen_audres_many_shot_{strategy}"+".csv")),
        model_type="llama",
        trial_flg=False,
        model_name_judge=model_name_judge,
        )

# %%
# output swallow
filename = "/Users/noro/Documents/Projects/XBRL_common_space_projection/data/3_processed/dataset_2310/downstream/few_shot_strategy_swallow/inf_rag_0125.csv"
pred_df = pd.read_csv(filename)
pred_df = pd.merge(pred_df,data_f_df,left_on="index_num",right_index=True)

for strategy in strategy_list:
    out_dir = PROJDIR / "data/3_processed/dataset_2310/downstream" / "few_shot_strategy_swallow" / f"batch_gen_audres_many_shot_{strategy}"
    out_dir.mkdir(parents=True,exist_ok=True)
    pred_df.query("custom_id.str.contains(@strategy)").to_csv(out_dir / f"batch_gen_audres_many_shot_{strategy}.csv",index=None)

model_name_judge = "gpt_4_turbo"
strategy_list = pred_df.custom_id.str.replace("request-","").str.split("_",expand=True).rename(columns={0:"strategy",1:"index_num"}).groupby("strategy").count().index.tolist()
for strategy in strategy_list:
    out_dir = PROJDIR / "data/3_processed/dataset_2310/downstream/few_shot_strategy_swallow" / f"batch_gen_audres_many_shot_{strategy}"
    out_dir.mkdir(parents=True, exist_ok=True)
    make_eval_batch_hal(
        out_dir=str(out_dir),
        out_ext=f"batch_gen_audres_many_shot_{strategy}"+"_",
        input_filename=str(out_dir / (f"batch_gen_audres_many_shot_{strategy}"+".csv")),
        model_type="llama",
        trial_flg=False,
        model_name_judge=model_name_judge,
        )
# %% #################################################################
#
# 20250230 gen cluster varilidity eval batch
#
######################################################################
GET_VECTOR_OBJ=get_train_data_vector()


def few_shot_prompt_near_and_clsfar_2(sr:pd.Series):
    instruction_1="監査担当者であるあなたは、監査上の検討事項を与えられたら、対応する監査上の対応事項を立案します。"
    instruction_2="例えば、次の検討事項が与えられました。\n\n##### 検討事項\n"
    instruction_3="\n\nこれに対応する監査上の対応事項は次のように立案されます。\n\n ##### 監査上の対応事項\n"
    instruction_4="以上のように監査上の対応事項を日本語文章で具体的に立案してください。"
    c_id=sr.id
    description_text=sr.description
    tar_vec=GET_VECTOR_OBJ.get_vecter(c_id)
    cls_label = GET_VECTOR_OBJ.get_cls_label(c_id)
    nearest_idx = GET_VECTOR_OBJ.get_nearest_id(tar_vec,k=1)
    nearest_idx_f = GET_VECTOR_OBJ.get_farthest_id_in_cls([tar_vec],cls_label,k=1)
    
    shot_description_text, shot_audit_res = GET_VECTOR_OBJ.train_instance(nearest_idx[0][0])
    shot_description_text_2, shot_audit_res_2 = GET_VECTOR_OBJ.train_instance(nearest_idx_f[0][0])
    
    sys_prompt=(instruction_1
        +"\n\n#### 例1\n\n"+instruction_2
        +shot_description_text+instruction_3+shot_audit_res
        +"\n\n#### 例2\n\n"+instruction_2
        +shot_description_text_2+instruction_3+shot_audit_res_2
        +instruction_4)
    usr_prompt="#### 検討事項\n"+description_text+"\n\n#### 監査上の対応事項"

    return sys_prompt,usr_prompt

model_name='llama_3.1_8b'
default_system_prompt="監査担当者であるあなたは、次の監査上の検討事項を与えられました。これに対応する監査上の対応事項を日本語文章で具体的に立案してください。"
batch_inf_file_generator_obj=batch_inf_file_generator(
            model_name=model_name,
            )

for index_num in tqdm(dict_df.index):
    sys_prompt,usr_prompt=few_shot_prompt_near_and_clsfar_2(dict_df.loc[index_num,:])
    itr_index_str="nf2cls_"+str(index_num)
    batch_inf_file_generator_obj.insert_inf_list_prompt(sys_prompt,usr_prompt,itr_index_str,max_tokens=1024, model_name=model_name)
batch_inf_file_generator_obj.print_sample()

out_filename=PROJDIR / "data/3_processed/dataset_2310/downstream" / "few_shot_strategy" /("batch_gen_audres_many_shot3_"+model_name+".jsonl")
batch_inf_file_generator_obj.export_list(out_filename)

# %% 2025/1/30 ###################################
#
# data lineageの作成
#
##################################################
import sys
sys.path.append(r'/Users/noro/Documents/Projects/XBRL_common_space_projection')

from src.data.libs.utils import DataLinageJson
import datetime
assertion_text = """
"""
processing_text = """
few_shot_prompt_near_and_clsfar_2の推論jsonl
"""
header_note_txt = """
"""
df = pd.DataFrame(batch_inf_file_generator_obj.inf_list)
file_path = out_filename
ts_str = datetime.datetime.fromtimestamp(os.path.getctime(file_path)).strftime('%Y-%m-%d %H:%M:%S')
ts_now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

DataLinageJson_obj = DataLinageJson(**{
    "create_date": f'{ts_str}',
    "check_date": f'{ts_now}',
    "size": f'{os.path.getsize(file_path):,}',
    "file_path": str(file_path),
    "reader": """
    with open(filename) as f:
        data_2 = f.readlines()
        data_2 = [json.loads(line) for line in data_2]
    """,
    "encoding": "utf-8",
    "input_data": {
        "dict_df":[str(PROJDIR / "data/3_processed/dataset_2310/downstream" / "2_intermediate/llm_proc" /"audit_res_markdown_eval.csv")],
        "vector_df":[str(PROJDIR / "data/3_processed/dataset_2310/downstream" / "all_data_mapping" /"dict_all_df_2407_v1012_vect_df.pkl")],
        "data_all_pivot":[str(PROJDIR / "data/3_processed/dataset_2310/downstream/all_data_mapping" / "data_all17k_pivot_2407_v1012_with_cls.csv")],
        "cls_label_df":[str(PROJDIR / "data/3_processed/dataset_2310/downstream/all_data_mapping/data_all17k_pivot_2407_v1012_with_cls.csv")],
        },
    "input_data_providing_func": {
        "vector_df": "get_train_data_vector",
        "data_all_pivot": "get_train_data_vector",
        "cls_label_df": "get_train_data_vector",
    },
    "index_name": df.index.name,
    "header": list(df.columns),
    "count": len(df),
    "unique_count_index": df.index.nunique(),
    "unique_count_header": df.describe(include='all').T['unique'].to_dict(),
    "example_rcd": df.iloc[0].to_dict(),
    "header_note": header_note_txt,
    "src": "data/ds01_06_kam_gen_shot_select_strategy.py",
    "assertion": "",
    "processing": processing_text,
    "note": ""
})
DataLinageJson_obj.save()

# %%
# %% #################################################################
#
# 20250131 knn-4 eval batch
#
######################################################################
GET_VECTOR_OBJ=get_train_data_vector()


def few_shot_prompt_kNN_4(sr:pd.Series):
    instruction_1="監査担当者であるあなたは、監査上の検討事項を与えられたら、対応する監査上の対応事項を立案します。"
    instruction_2="例えば、次の検討事項が与えられました。\n\n##### 検討事項\n"
    instruction_3="\n\nこれに対応する監査上の対応事項は次のように立案されます。\n\n ##### 監査上の対応事項\n"
    instruction_4="以上のように監査上の対応事項を日本語文章で具体的に立案してください。"
    c_id=sr.id
    description_text=sr.description
    tar_vec=GET_VECTOR_OBJ.get_vecter(c_id)
    nearest_idx=GET_VECTOR_OBJ.get_nearest_id(tar_vec,k=5)
    shot_description_text, shot_audit_res = GET_VECTOR_OBJ.train_instance(nearest_idx[0][0])
    shot_description_text_2, shot_audit_res_2 = GET_VECTOR_OBJ.train_instance(nearest_idx[1][0])
    shot_description_text_3, shot_audit_res_3 = GET_VECTOR_OBJ.train_instance(nearest_idx[2][0])
    shot_description_text_4, shot_audit_res_4 = GET_VECTOR_OBJ.train_instance(nearest_idx[3][0])
    #shot_description_text_5, shot_audit_res_5 = GET_VECTOR_OBJ.train_instance(nearest_idx[4][0])
    
    sys_prompt=(instruction_1
        +"\n\n#### 例1\n\n"+instruction_2
        +shot_description_text+instruction_3+shot_audit_res
        +"\n\n#### 例2\n\n"+instruction_2
        +shot_description_text_2+instruction_3+shot_audit_res_2
        +"\n\n#### 例3\n\n"+instruction_2
        +shot_description_text_3+instruction_3+shot_audit_res_3
        +"\n\n#### 例4\n\n"+instruction_2
        +shot_description_text_4+instruction_3+shot_audit_res_4
        #+"\n\n#### 例5\n\n"+instruction_2
        #+shot_description_text_5+instruction_3+shot_audit_res_5
        +"\n\n"+instruction_4)
    usr_prompt="#### 検討事項\n"+description_text+"\n\n#### 監査上の対応事項"

    return sys_prompt,usr_prompt


def few_shot_prompt_inv_kNN_4(sr:pd.Series):
    instruction_1="監査担当者であるあなたは、監査上の検討事項を与えられたら、対応する監査上の対応事項を立案します。"
    instruction_2="例えば、次の検討事項が与えられました。\n\n##### 検討事項\n"
    instruction_3="\n\nこれに対応する監査上の対応事項は次のように立案されます。\n\n ##### 監査上の対応事項\n"
    instruction_4="以上のように監査上の対応事項を日本語文章で具体的に立案してください。"
    c_id=sr.id
    description_text=sr.description
    tar_vec=GET_VECTOR_OBJ.get_vecter(c_id)
    farthest_idx=GET_VECTOR_OBJ.get_farthest_id([tar_vec],k=1)
    smp_1_vec=GET_VECTOR_OBJ.get_vecter(farthest_idx[0][0])
    farthest_idx_2=GET_VECTOR_OBJ.get_farthest_id([tar_vec,smp_1_vec],k=1)
    smp_2_vec=GET_VECTOR_OBJ.get_vecter(farthest_idx_2[0][0])
    farthest_idx_3=GET_VECTOR_OBJ.get_farthest_id([tar_vec,smp_1_vec,smp_2_vec],k=1)
    smp_3_vec=GET_VECTOR_OBJ.get_vecter(farthest_idx_3[0][0])
    farthest_idx_4=GET_VECTOR_OBJ.get_farthest_id([tar_vec,smp_1_vec,smp_2_vec,smp_3_vec],k=1)
    smp_4_vec=GET_VECTOR_OBJ.get_vecter(farthest_idx_4[0][0])
    farthest_idx_5=GET_VECTOR_OBJ.get_farthest_id([tar_vec,smp_1_vec,smp_2_vec,smp_3_vec,smp_4_vec],k=1)

    shot_description_text, shot_audit_res = GET_VECTOR_OBJ.train_instance(farthest_idx[0][0])
    shot_description_text_2, shot_audit_res_2 = GET_VECTOR_OBJ.train_instance(farthest_idx_2[0][0])
    shot_description_text_3, shot_audit_res_3 = GET_VECTOR_OBJ.train_instance(farthest_idx_3[0][0])
    shot_description_text_4, shot_audit_res_4 = GET_VECTOR_OBJ.train_instance(farthest_idx_4[0][0])
    #shot_description_text_5, shot_audit_res_5 = GET_VECTOR_OBJ.train_instance(farthest_idx_5[0][0])
    
    sys_prompt=(instruction_1
        +"\n\n#### 例1\n\n"+instruction_2
        +shot_description_text+instruction_3+shot_audit_res
        +"\n\n#### 例2\n\n"+instruction_2
        +shot_description_text_2+instruction_3+shot_audit_res_2
        +"\n\n#### 例3\n\n"+instruction_2
        +shot_description_text_3+instruction_3+shot_audit_res_3
        +"\n\n#### 例4\n\n"+instruction_2
        +shot_description_text_4+instruction_3+shot_audit_res_4
        #+"\n\n#### 例5\n\n"+instruction_2
        #+shot_description_text_5+instruction_3+shot_audit_res_5
        +"\n\n"+instruction_4)
    usr_prompt="#### 検討事項\n"+description_text+"\n\n#### 監査上の対応事項"

    return sys_prompt,usr_prompt


model_name='llama_3.1_8b'
batch_inf_file_generator_obj=batch_inf_file_generator(
            model_name=model_name,
            )
for index_num in tqdm(dict_df.index):
    sys_prompt,usr_prompt=few_shot_prompt_inv_kNN_4(dict_df.loc[index_num,:])
    itr_index_str="nf2cls_"+str(index_num)
    batch_inf_file_generator_obj.insert_inf_list_prompt(sys_prompt,usr_prompt,itr_index_str,max_tokens=1024, model_name=model_name)
batch_inf_file_generator_obj.print_sample()

#out_filename=PROJDIR / "data/3_processed/dataset_2310/downstream" / "few_shot_strategy" /("batch_gen_audres_many_shot4_"+model_name+".jsonl")
#batch_inf_file_generator_obj.export_list(out_filename)


model_name='llama_3.1_8b'
batch_inf_file_generator_obj=batch_inf_file_generator(
            model_name=model_name,
            )
for index_num in tqdm(dict_df.index):
    sys_prompt,usr_prompt=few_shot_prompt_kNN_4(dict_df.loc[index_num,:])
    itr_index_str="knn4_"+str(index_num)
    batch_inf_file_generator_obj.insert_inf_list_prompt(sys_prompt,usr_prompt,itr_index_str,max_tokens=1024, model_name=model_name)
batch_inf_file_generator_obj.print_sample()

out_filename=PROJDIR / "data/3_processed/dataset_2310/downstream" / "few_shot_strategy" /("batch_gen_audres_many_shot5_"+model_name+".jsonl")
batch_inf_file_generator_obj.export_list(out_filename)

# %% 2025/1/31 ###################################
#
# data lineageの作成
#
##################################################
import sys
sys.path.append(r'/Users/noro/Documents/Projects/XBRL_common_space_projection')

from src.data.libs.utils import DataLinageJson
import datetime
assertion_text = """
"""
processing_text = """
few_shot_prompt_kNN_4の推論jsonl
"""
header_note_txt = """
"""
df = pd.DataFrame(batch_inf_file_generator_obj.inf_list)
file_path = out_filename
ts_str = datetime.datetime.fromtimestamp(os.path.getctime(file_path)).strftime('%Y-%m-%d %H:%M:%S')
ts_now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

DataLinageJson_obj = DataLinageJson(**{
    "create_date": f'{ts_str}',
    "check_date": f'{ts_now}',
    "size": f'{os.path.getsize(file_path):,}',
    "file_path": str(file_path),
    "reader": """
    with open(filename) as f:
        data_2 = f.readlines()
        data_2 = [json.loads(line) for line in data_2]
    """,
    "encoding": "utf-8",
    "input_data": {
        "dict_df":[str(PROJDIR / "data/3_processed/dataset_2310/downstream" / "2_intermediate/llm_proc" /"audit_res_markdown_eval.csv")],
        "vector_df":[str(PROJDIR / "data/3_processed/dataset_2310/downstream" / "all_data_mapping" /"dict_all_df_2407_v1012_vect_df.pkl")],
        "data_all_pivot":[str(PROJDIR / "data/3_processed/dataset_2310/downstream/all_data_mapping" / "data_all17k_pivot_2407_v1012_with_cls.csv")],
        },
    "input_data_providing_func": {
        "vector_df": "get_train_data_vector",
        "data_all_pivot": "get_train_data_vector",
        },
    "index_name": df.index.name,
    "header": list(df.columns),
    "count": len(df),
    "unique_count_index": df.index.nunique(),
    "unique_count_header": df.describe(include='all').T['unique'].to_dict(),
    "example_rcd": df.iloc[0].to_dict(),
    "header_note": header_note_txt,
    "src": "data/ds01_06_kam_gen_shot_select_strategy.py",
    "assertion": "",
    "processing": processing_text,
    "note": ""
})
DataLinageJson_obj.save()

# %% #################################################################
#
# 20250205 rand4, near far 4 for Qwen2
#
############################################################################

GET_VECTOR_OBJ=get_train_data_vector()

def few_shot_prompt_near_and_far_4_cor(sr:pd.Series):
    instruction_1="監査担当者であるあなたは、監査上の検討事項を与えられたら、対応する監査上の対応事項を立案します。"
    instruction_2="例えば、次の検討事項が与えられました。\n\n##### 検討事項\n"
    instruction_3="\n\nこれに対応する監査上の対応事項は次のように立案されます。\n\n ##### 監査上の対応事項\n"
    instruction_4="以上のように監査上の対応事項を日本語文章で具体的に立案してください。"
    c_id=sr.id
    description_text=sr.description
    tar_vec=GET_VECTOR_OBJ.get_vecter(c_id)
    nearest_idx=GET_VECTOR_OBJ.get_nearest_id(tar_vec,k=1)
    
    smp_1_vec=GET_VECTOR_OBJ.get_vecter(nearest_idx[0][0])
    farthest_idx_2=GET_VECTOR_OBJ.get_farthest_id([tar_vec,smp_1_vec],k=1)
    smp_2_vec=GET_VECTOR_OBJ.get_vecter(farthest_idx_2[0][0])
    farthest_idx_3=GET_VECTOR_OBJ.get_farthest_id([tar_vec,smp_1_vec,smp_2_vec],k=1)
    smp_3_vec=GET_VECTOR_OBJ.get_vecter(farthest_idx_3[0][0])
    farthest_idx_4=GET_VECTOR_OBJ.get_farthest_id([tar_vec,smp_1_vec,smp_2_vec,smp_3_vec],k=1)
    smp_4_vec=GET_VECTOR_OBJ.get_vecter(farthest_idx_4[0][0])
    farthest_idx_5=GET_VECTOR_OBJ.get_farthest_id([tar_vec,smp_1_vec,smp_2_vec,smp_3_vec,smp_4_vec],k=1)

    shot_description_text, shot_audit_res = GET_VECTOR_OBJ.train_instance(nearest_idx[0][0])
    shot_description_text_2, shot_audit_res_2 = GET_VECTOR_OBJ.train_instance(farthest_idx_2[0][0])
    shot_description_text_3, shot_audit_res_3 = GET_VECTOR_OBJ.train_instance(farthest_idx_3[0][0])
    shot_description_text_4, shot_audit_res_4 = GET_VECTOR_OBJ.train_instance(farthest_idx_4[0][0])
    #shot_description_text_5, shot_audit_res_5 = GET_VECTOR_OBJ.train_instance(farthest_idx_5[0][0])
    
    sys_prompt=(instruction_1
        +"\n\n#### 例1\n\n"+instruction_2
        +shot_description_text+instruction_3+shot_audit_res
        +"\n\n#### 例2\n\n"+instruction_2
        +shot_description_text_2+instruction_3+shot_audit_res_2
        +"\n\n#### 例3\n\n"+instruction_2
        +shot_description_text_3+instruction_3+shot_audit_res_3
        +"\n\n#### 例4\n\n"+instruction_2
        +shot_description_text_4+instruction_3+shot_audit_res_4
        #+"\n\n#### 例5\n\n"+instruction_2
        #+shot_description_text_5+instruction_3+shot_audit_res_5
        +instruction_4)
    usr_prompt="#### 検討事項\n"+description_text+"\n\n#### 監査上の対応事項"

    return sys_prompt,usr_prompt


def few_shot_prompt_rand_4(sr:pd.Series):
    instruction_1="監査担当者であるあなたは、監査上の検討事項を与えられたら、対応する監査上の対応事項を立案します。"
    instruction_2="例えば、次の検討事項が与えられました。\n\n##### 検討事項\n"
    instruction_3="\n\nこれに対応する監査上の対応事項は次のように立案されます。\n\n ##### 監査上の対応事項\n"
    instruction_4="以上のように監査上の対応事項を日本語文章で具体的に立案してください。"
    c_id=sr.id
    description_text=sr.description
    tar_vec=GET_VECTOR_OBJ.get_vecter(c_id)
    #nearest_idx=GET_VECTOR_OBJ.get_farthest_id(tar_vec,k=5)
    random_idx=np.random.choice(GET_VECTOR_OBJ.data_train.index,5)
    shot_description_text, shot_audit_res = GET_VECTOR_OBJ.train_instance(random_idx[0])
    shot_description_text_2, shot_audit_res_2 = GET_VECTOR_OBJ.train_instance(random_idx[1])
    shot_description_text_3, shot_audit_res_3 = GET_VECTOR_OBJ.train_instance(random_idx[2])
    shot_description_text_4, shot_audit_res_4 = GET_VECTOR_OBJ.train_instance(random_idx[3])
    #shot_description_text_5, shot_audit_res_5 = GET_VECTOR_OBJ.train_instance(random_idx[4])
    
    sys_prompt=(instruction_1
        +"\n\n#### 例1\n\n"+instruction_2
        +shot_description_text+instruction_3+shot_audit_res
        +"\n\n#### 例2\n\n"+instruction_2
        +shot_description_text_2+instruction_3+shot_audit_res_2
        +"\n\n#### 例3\n\n"+instruction_2
        +shot_description_text_3+instruction_3+shot_audit_res_3
        +"\n\n#### 例4\n\n"+instruction_2
        +shot_description_text_4+instruction_3+shot_audit_res_4
        #+"\n\n#### 例5\n\n"+instruction_2
        #+shot_description_text_5+instruction_3+shot_audit_res_5
        +instruction_4)
    usr_prompt="#### 検討事項\n"+description_text+"\n\n#### 監査上の対応事項"

    return sys_prompt,usr_prompt




model_name='llama_3.1_8b'
batch_inf_file_generator_obj=batch_inf_file_generator(
            model_name=model_name,
            )
for index_num in tqdm(dict_df.index):
    sys_prompt,usr_prompt=few_shot_prompt_near_and_far_4_cor(dict_df.loc[index_num,:])
    itr_index_str="nf4cor_"+str(index_num)
    batch_inf_file_generator_obj.insert_inf_list_prompt(sys_prompt,usr_prompt,itr_index_str,max_tokens=1024, model_name=model_name)
batch_inf_file_generator_obj.print_sample()

for index_num in tqdm(dict_df.index):
    sys_prompt,usr_prompt=few_shot_prompt_rand_4(dict_df.loc[index_num,:])
    itr_index_str="rand4_"+str(index_num)
    batch_inf_file_generator_obj.insert_inf_list_prompt(sys_prompt,usr_prompt,itr_index_str,max_tokens=1024, model_name=model_name)
batch_inf_file_generator_obj.print_sample()

out_filename=PROJDIR / "data/3_processed/dataset_2310/downstream" / "inf_data_few_shot_strategy" /("batch_gen_audres_many_shot6_"+model_name+".jsonl")
batch_inf_file_generator_obj.export_list(out_filename)

# %% 2025/1/31 ###################################
#
# data lineageの作成
#
##################################################
import sys
sys.path.append(r'/Users/noro/Documents/Projects/XBRL_common_space_projection')

from src.data.libs.utils import DataLinageJson
import datetime
assertion_text = """
"""
processing_text = """
few_shot_prompt_near_and_far_4_corの推論jsonl
few_shot_prompt_rand_4の推論jsonl
"""
header_note_txt = """
"""
df = pd.DataFrame(batch_inf_file_generator_obj.inf_list)
file_path = out_filename
ts_str = datetime.datetime.fromtimestamp(os.path.getctime(file_path)).strftime('%Y-%m-%d %H:%M:%S')
ts_now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

DataLinageJson_obj = DataLinageJson(**{
    "create_date": f'{ts_str}',
    "check_date": f'{ts_now}',
    "size": f'{os.path.getsize(file_path):,}',
    "file_path": str(file_path),
    "reader": """
    with open(filename) as f:
        data_2 = f.readlines()
        data_2 = [json.loads(line) for line in data_2]
    """,
    "encoding": "utf-8",
    "input_data": {
        "dict_df":[str(PROJDIR / "data/3_processed/dataset_2310/downstream" / "2_intermediate/llm_proc" /"audit_res_markdown_eval.csv")],
        "vector_df":[str(PROJDIR / "data/3_processed/dataset_2310/downstream" / "all_data_mapping" /"dict_all_df_2407_v1012_vect_df.pkl")],
        "data_all_pivot":[str(PROJDIR / "data/3_processed/dataset_2310/downstream/all_data_mapping" / "data_all17k_pivot_2407_v1012_with_cls.csv")],
        },
    "input_data_providing_func": {
        "vector_df": "get_train_data_vector",
        "data_all_pivot": "get_train_data_vector",
        },
    "index_name": df.index.name,
    "header": list(df.columns),
    "count": len(df),
    "unique_count_index": df.index.nunique(),
    "unique_count_header": df.describe(include='all').T['unique'].to_dict(),
    "example_rcd": df.iloc[0].to_dict(),
    "header_note": header_note_txt,
    "src": "data/ds01_06_kam_gen_shot_select_strategy.py",
    "assertion": "",
    "processing": processing_text,
    "note": ""
})
DataLinageJson_obj.save()
# %% gen eval batch 6

# NOTE 一部はds01_06_make_eval_batch_gpt4で作成している

from libs.kam_evaluation import *
# input

# customer_idが推論時に保存し忘れのため追加
filename=PROJDIR / "data/3_processed/dataset_2310/downstream" / "inf_data_few_shot_strategy" /("batch_gen_audres_many_shot6_llama_3.1_8b.jsonl")
with open(filename) as f:
    data = f.readlines()
    data = [json.loads(line) for line in data]

# %%
data_df = pd.DataFrame(data)
# output qwen
filename = "/Users/noro/Documents/Projects/XBRL_common_space_projection/data/3_processed/dataset_2310/downstream/few_shot_strategy_qwen2/inf_rag_6_0205.csv"
pred_df = pd.read_csv(filename)
pred_df = pd.merge(pred_df,data_df,left_on="index_num",right_index=True)
model_name_judge = "gpt_4_turbo"

strategy_list = pred_df.custom_id.str.replace("request-","").str.split("_",expand=True).rename(columns={0:"strategy",1:"index_num"}).groupby("strategy").count().index.tolist()

for strategy in strategy_list:
    out_dir = PROJDIR / "data/3_processed/dataset_2310/downstream" / "few_shot_strategy_qwen2" / f"batch_gen_audres_many_shot_{strategy}"
    out_dir.mkdir(parents=True,exist_ok=True)
    pred_df.query("custom_id.str.contains(@strategy)").to_csv(out_dir / f"batch_gen_audres_many_shot_{strategy}.csv",index=None)

model_name_judge = "gpt_4_turbo"
strategy_list = pred_df.custom_id.str.replace("request-","").str.split("_",expand=True).rename(columns={0:"strategy",1:"index_num"}).groupby("strategy").count().index.tolist()
for strategy in strategy_list:
    out_dir = PROJDIR / "data/3_processed/dataset_2310/downstream/few_shot_strategy_qwen2" / f"batch_gen_audres_many_shot_{strategy}"
    out_dir.mkdir(parents=True, exist_ok=True)
    make_eval_batch(
        out_dir=str(out_dir),
        out_ext=f"batch_gen_audres_many_shot_{strategy}"+"_",
        input_filename=str(out_dir / (f"batch_gen_audres_many_shot_{strategy}"+".csv")),
        model_type="llama",
        trial_flg=False,
        model_name_judge=model_name_judge,
        )
# %%
