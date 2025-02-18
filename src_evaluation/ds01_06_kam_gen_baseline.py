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
    default_system_prompt="監査担当者であるあなたは、次の監査上の検討事項を与えられました。これに対応する監査上の対応事項を日本語文章で具体的に立案してください。"
    batch_inf_file_generator_obj=batch_inf_file_generator(
                model_name=model_name,
                )

    for index_num in tqdm(dict_df.index):
        #description_text=dict_df.loc[index_num,'description']
        sys_prompt,usr_prompt=prompt_gen_func(dict_df.loc[index_num,:])
        #sys_prompt=default_system_prompt
        #usr_prompt=description_text
        itr_index_str=str(index_num)
        batch_inf_file_generator_obj.insert_inf_list_prompt(sys_prompt,usr_prompt,itr_index_str,max_tokens=1024, model_name=model_name)

    #out_filename=PROJDIR / "data/3_processed/dataset_2310/downstream" / "baseline" /("batch_gen_audres_"+model_name+".jsonl")
    batch_inf_file_generator_obj.export_list(out_filename)
    batch_inf_file_generator_obj.print_sample()
    
#model_name="llama_3.1_8b"
#out_filename=PROJDIR / "data/3_processed/dataset_2310/downstream" / "baseline" /("batch_gen_audres_"+model_name+".jsonl")
#make_batch(dict_df,out_filename,model_name=model_name,prompt_gen_func=default_prompt)

# %% plane gpt 4o mini
default_system_prompt="監査担当者であるあなたは、次の監査上の検討事項を与えられました。これに対応する監査上の対応事項を日本語文章で具体的に立案してください。"
batch_inf_file_generator_obj=batch_inf_file_generator(
            model_name='gpt_4o_mini',
            )

for index_num in dict_df.index:
    description_text=dict_df.loc[index_num,'description']
    sys_prompt=default_system_prompt
    usr_prompt=description_text
    itr_index_str=str(index_num)
    batch_inf_file_generator_obj.insert_inf_list_prompt(sys_prompt,usr_prompt,itr_index_str)

out_filename=PROJDIR / "data/3_processed/dataset_2310/downstream" / "baseline" /"aud_res_gpt_4o_mini.jsonl"
#batch_inf_file_generator_obj.export_list(out_filename)
#batch_inf_file_generator_obj.print_sample()



# %% gpt-4

default_system_prompt="監査担当者であるあなたは、次の監査上の検討事項を与えられました。これに対応する監査上の対応事項を日本語文章で具体的に立案してください。"
batch_inf_file_generator_obj=batch_inf_file_generator(
            model_name='gpt_4',
            )

for index_num in dict_df.index:
    description_text=dict_df.loc[index_num,'description']
    sys_prompt=default_system_prompt
    usr_prompt=description_text
    itr_index_str=str(index_num)
    batch_inf_file_generator_obj.insert_inf_list_prompt(sys_prompt,usr_prompt,itr_index_str)

out_filename=PROJDIR / "data/3_processed/dataset_2310/downstream" / "baseline" /"aud_res_gpt_4.jsonl"
#batch_inf_file_generator_obj.export_list(out_filename)
#batch_inf_file_generator_obj.print_sample()

# %% gpt-4o

default_system_prompt="監査担当者であるあなたは、次の監査上の検討事項を与えられました。これに対応する監査上の対応事項を日本語文章で具体的に立案してください。"
batch_inf_file_generator_obj=batch_inf_file_generator(
            model_name='gpt_4o',
            )

for index_num in dict_df.index:
    description_text=dict_df.loc[index_num,'description']
    sys_prompt=default_system_prompt
    usr_prompt=description_text
    itr_index_str=str(index_num)
    batch_inf_file_generator_obj.insert_inf_list_prompt(sys_prompt,usr_prompt,itr_index_str)

out_filename=PROJDIR / "data/3_processed/dataset_2310/downstream" / "baseline" /"aud_res_gpt_4o.jsonl"
#batch_inf_file_generator_obj.export_list(out_filename)
#batch_inf_file_generator_obj.print_sample()


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
            sim_list = []
            for vector in vector_list:
                sim_list.append(cosine_similarity(vector, row))
            # calc max
            similarities.append((index, max(sim_list)))
        top_matches = sorted(similarities, key=lambda x: x[1], reverse=False)[:k]
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

# %% eval validation sample 項目反応理論?

GET_VECTOR_OBJ=get_train_data_vector()

nearest_score_list=[]
for index_num in tqdm(dict_df.index):
    c_id=dict_df.loc[index_num,'id']
    #'S100T3LA_FilingDateInstant_Row1Member'
    #description_text=sr.description
    tar_vec=GET_VECTOR_OBJ.get_vecter(c_id)
    nearest_idx=GET_VECTOR_OBJ.get_nearest_id(tar_vec)
    #shot_description_text, shot_audit_res = GET_VECTOR_OBJ.train_instance(nearest_idx[0][0])
    ant_t={
        "index_num":index_num,
        "c_id":c_id,
        "nearest_id":nearest_idx[0][0][0],
        "nearest_score":nearest_idx[0][0][1]
        }
    nearest_idx[0][0][1]
    nearest_score_list.append(ant_t)

#pd.DataFrame(nearest_score_list).to_csv(PROJDIR / "data/3_processed/dataset_2310/downstream" / "eval_prep" / "nearest_score.csv",index=None)


# %%
def few_shot_prompt(sr:pd.Series):
    instruction_1="監査担当者であるあなたは、監査上の検討事項を与えられたら、対応する監査上の対応事項を立案します。"
    instruction_2="例えば、次の検討事項が与えられました。\n\n#### 検討事項\n"
    instruction_3="\n\nこれに対応する監査上の対応事項は次のように立案されます。\n\n #### 監査上の対応事項\n"
    instruction_4="以上のように監査上の対応事項を日本語文章で具体的に立案してください。"
    c_id=sr.id
    description_text=sr.description
    tar_vec=GET_VECTOR_OBJ.get_vecter(c_id)
    nearest_idx=GET_VECTOR_OBJ.get_nearest_id(tar_vec)
    shot_description_text, shot_audit_res = GET_VECTOR_OBJ.train_instance(nearest_idx[0][0])
    sys_prompt=instruction_1+instruction_2+shot_description_text+instruction_3+shot_audit_res+instruction_4
    usr_prompt="#### 検討事項\n"+description_text+"\n\n#### 監査上の対応事項"

    #sys_prompt=default_system_prompt
    #usr_prompt=description_text
    return sys_prompt,usr_prompt

GET_VECTOR_OBJ=get_train_data_vector()

model_name="llama_3.1_8b"
out_filename=PROJDIR / "data/3_processed/dataset_2310/downstream" / "baseline" /("batch_gen_audres_2shot_"+model_name+".jsonl")
make_batch(dict_df,out_filename,model_name=model_name,prompt_gen_func=few_shot_prompt)

model_name="gpt_4o_mini"
out_filename=PROJDIR / "data/3_processed/dataset_2310/downstream" / "baseline" /("batch_gen_audres_2shot_"+model_name+".jsonl")
#make_batch(dict_df,out_filename,model_name=model_name,prompt_gen_func=few_shot_prompt)
# %%
model_name="gpt_4"
out_filename=PROJDIR / "data/3_processed/dataset_2310/downstream" / "baseline" /("batch_gen_audres_1shot_"+model_name+".jsonl")
make_batch(dict_df,out_filename,model_name=model_name,prompt_gen_func=few_shot_prompt)

# %%

model_name="gpt_4o"
out_filename=PROJDIR / "data/3_processed/dataset_2310/downstream" / "baseline" /("batch_gen_audres_1shot_"+model_name+".jsonl")
make_batch(dict_df,out_filename,model_name=model_name,prompt_gen_func=few_shot_prompt)

# %%


# %% (eval) ###############################################################################
#
# 1. モデルの出力結果を評価するためのデータセットを作成する
#
###########################################################################################
# %%
proc_text = """
モデルの出力結果を評価するためのデータセットを作成する 
1 正解をそのまま評価

網羅性 gpt-4o-mini
2 正解の一部を追加して作成

具体性 gpt-4o
出力の共通点を要約

関連性 gpt-4o
3 無関係な出力の追加

"""



# %% 1. 正解をそのまま評価 and 2. 正解を追加
import random
random.seed(0)

def validate_eval_addallans_prompt(sr:pd.Series):
    instruction_1="監査担当者であるあなたは、次の監査上の検討事項を与えられました。"
    instruction_2="これに対応する監査上の対応事項を日本語文章で具体的に立案してください。"

    description_text=sr.description
    proc_ans=sr.output
    #hint_text="#### ヒント\n次は対応事項の候補の一部です。\n * "+"\n * ".join(proc_smp_list)
    hint_text="#### ヒント\nただし、次の対応事項を回答に含めてください。\n" + proc_ans
    
    sys_prompt=instruction_1+"\n\n#### 検討事項\n"+description_text
    usr_prompt=instruction_2+"\n\n"+hint_text+"\n\n#### 監査上の対応事項"
    return sys_prompt,usr_prompt


def validate_eval_addans1_prompt(sr:pd.Series):
    instruction_1="監査担当者であるあなたは、次の監査上の検討事項を与えられました。"
    instruction_2="これに対応する監査上の対応事項を日本語文章で具体的に立案してください。"

    description_text=sr.description
    proc_list=sr.prep_output
    s =1
    proc_list=[list(item_dict.values())[0] for item_dict in proc_list]
    proc_smp_list=random.sample(proc_list,s)
    #hint_text="#### ヒント\n次は対応事項の候補の一部です。\n * "+"\n * ".join(proc_smp_list)
    hint_text="#### ヒント\nただし、次の対応事項を回答に含めてください。\n * "+"\n * ".join(proc_smp_list)
    
    sys_prompt=instruction_1+"\n\n#### 検討事項\n"+description_text
    usr_prompt=instruction_2+"\n\n"+hint_text+"\n\n#### 監査上の対応事項"
    return sys_prompt,usr_prompt
    

def validate_eval_addans2_prompt(sr:pd.Series):
    instruction_1="監査担当者であるあなたは、次の監査上の検討事項を与えられました。"
    instruction_2="これに対応する監査上の対応事項を日本語文章で具体的に立案してください。"

    description_text=sr.description
    proc_list=sr.prep_output
    s =2
    proc_list=[list(item_dict.values())[0] for item_dict in proc_list]
    proc_smp_list=random.sample(proc_list,s)
    #hint_text="#### ヒント\n次は対応事項の候補の一部です。\n * "+"\n * ".join(proc_smp_list)
    hint_text="#### ヒント\nただし、次の対応事項を回答に含めてください。\n * "+"\n * ".join(proc_smp_list)
    
    sys_prompt=instruction_1+"\n\n#### 検討事項\n"+description_text
    usr_prompt=instruction_2+"\n\n"+hint_text+"\n\n#### 監査上の対応事項"
    return sys_prompt,usr_prompt

# 評価データ audet_resのmarkdown
filename=PROJDIR / "data/3_processed/dataset_2310/downstream" / "2_intermediate/llm_proc" /"audit_res_markdown_eval.csv"
dict_df=pd.read_csv(filename,index_col=None,dtype=str).set_index('index_num')

# %% 監査手続をリスト化
filename_openai_rst_proc=PROJDIR / "data/3_processed/dataset_2310/downstream" / "eval_prep" /"eval_extracted_process_from_ans_output.jsonl"
response_list_df=pd.DataFrame(get_results_openai_batch(filename_openai_rst=filename_openai_rst_proc)).query("status!='Failed'")
# index復元
response_list_df=response_list_df.assign(
    index_num_df=response_list_df.index_num.str.replace("request-","")
    )
# マージのためにカラム名変更
response_list_df = response_list_df.set_index('index_num_df').rename(
    columns={'status':'prep_status','output':'prep_output'}
    )
# マージ
dict_df_con=pd.merge(dict_df,response_list_df[['prep_output']],left_index=True,right_index=True,how='left')


# openaiバッチjsonl作成
model_name="gpt_4o"
out_filename=PROJDIR / "data/3_processed/dataset_2310/downstream" / "eval_data" /("aud_res_"+model_name+"_add_ans_1.jsonl")
make_batch(dict_df_con,out_filename,model_name=model_name,prompt_gen_func=validate_eval_addans1_prompt)

model_name="gpt_4o"
out_filename=PROJDIR / "data/3_processed/dataset_2310/downstream" / "eval_data" /("aud_res_"+model_name+"_add_ans_2.jsonl")
make_batch(dict_df_con,out_filename,model_name=model_name,prompt_gen_func=validate_eval_addans2_prompt)

model_name="gpt_4o"
out_filename=PROJDIR / "data/3_processed/dataset_2310/downstream" / "eval_data" /("aud_res_"+model_name+"_add_ans_all.jsonl")
make_batch(dict_df_con,out_filename,model_name=model_name,prompt_gen_func=validate_eval_addallans_prompt)

# 比率は0.2でも影響が大きいため1個,2個, すべてに変更
#for itr_r in [0.2,0.4,0.6,0.8]:
#    dict_df_con['r']=itr_r
#    model_name="llama_3.1_8b"
#    out_filename=PROJDIR / "data/3_processed/dataset_2310/downstream" / "eval_data" /("aud_res_"+model_name+"_add_ans_"+str(itr_r)+".jsonl")
#    make_batch(dict_df_con,out_filename,model_name=model_name,prompt_gen_func=validate_eval_addans_prompt)



# %% (eval 具体性) general answer generation
# %% step 1
def validate_eval_abstract_prompt(sr:pd.Series):
    """
        同じ論点のトピックの複数の監査手続を与えて一般化してもらう
        追加するサンプルの近さを変えて、評価
        出力の共通点を要約
    """

    instruction_1="次のいくつかの監査上の検討事項を要約し、一般化した1つの検討事項の事例にしてください"
    instruction_2="\n\n#### 検討事項\n"
    
    c_id=sr.id
    description_text=sr.description
    ans_text=sr.audit_res
    tar_vec=GET_VECTOR_VALID_OBJ.get_vecter(c_id)
    nearest_idx=GET_VECTOR_VALID_OBJ.get_nearest_id(tar_vec,k=2)
    shot_description_text, shot_audit_res = GET_VECTOR_VALID_OBJ.validation_instance(nearest_idx[1][0])
    sys_prompt=instruction_1
    usr_prompt=instruction_2+description_text+"\n\n"+shot_description_text+"\n\n#### 一般化した検討事項の事例"

    return sys_prompt,usr_prompt

GET_VECTOR_VALID_OBJ=get_validation_data_vector()

model_name="gpt_4o"
out_filename=PROJDIR / "data/3_processed/dataset_2310/downstream" / "eval_data" /("prep_eval_abs_"+model_name+".jsonl")
make_batch(dict_df,out_filename,model_name=model_name,prompt_gen_func=validate_eval_abstract_prompt)

# %%
def validate_eval_abstract_prompt_2(sr:pd.Series):
    """
        同じ論点のトピックの複数の監査手続を与えて一般化してもらう
        追加するサンプルの近さを変えて、評価
        出力の共通点を要約
    """

    instruction_1="次のいくつかの監査上の検討事項から共通部分を要約し、一般化した1つの検討事項の事例にしてください。\n回答は「一般化した1つの検討事項の事例」のみ出力してください。"
    instruction_2="\n\n#### 検討事項\n"
    
    c_id=sr.id
    description_text=sr.description
    ans_text=sr.audit_res
    tar_vec=GET_VECTOR_VALID_OBJ.get_vecter(c_id)
    nearest_idx=GET_VECTOR_VALID_OBJ.get_nearest_id(tar_vec,k=4)
    shot_description_text, shot_audit_res = GET_VECTOR_VALID_OBJ.validation_instance(nearest_idx[1][0])
    shot_description_text_2, shot_audit_res = GET_VECTOR_VALID_OBJ.validation_instance(nearest_idx[2][0])
    shot_description_text_3, shot_audit_res = GET_VECTOR_VALID_OBJ.validation_instance(nearest_idx[3][0])
    
    
    sys_prompt=instruction_1
    usr_prompt=(instruction_2
        +description_text
        +"\n\n"+shot_description_text
        +"\n\n"+shot_description_text_2
        +"\n\n"+shot_description_text_3
        +"\n\n#### 一般化した検討事項の事例")

    return sys_prompt,usr_prompt

model_name="gpt_4o"
out_filename=PROJDIR / "data/3_processed/dataset_2310/downstream" / "eval_data" /("prep_eval_abs_k4_"+model_name+".jsonl")
make_batch(dict_df,out_filename,model_name=model_name,prompt_gen_func=validate_eval_abstract_prompt_2)


# %% step2
# k1
DOWNSTREAM_PATH = PROJDIR/"data/3_processed/dataset_2310/downstream"
filename_openai_rst = DOWNSTREAM_PATH / "eval_data/prep_eval_abs_gpt_4o_output.jsonl"
ans_list = get_results_openai_batch(filename_openai_rst,json=False)
ans_df = pd.DataFrame(ans_list)
ans_df.index_num = ans_df.index_num.str.replace("request-","")#.astype(int)
dict_df_with_dummy = pd.merge(dict_df,ans_df.rename(columns={'output':'dummy_description'}),left_on='index_num',right_on='index_num',how='left')


def delete_title(text):
    lines = text.split("\n")
    lines = [line for line in lines if not "検討事項の事例" in line]
    return "\n".join(lines)
dict_df_with_dummy.dummy_description = dict_df_with_dummy.dummy_description.apply(delete_title)


def default_prompt_dummy_k1(sr:pd.Series):
    default_system_prompt="監査担当者であるあなたは、次の監査上の検討事項を与えられました。これに対応する監査上の対応事項を日本語文章で具体的に立案してください。"
    description_text=sr.dummy_description
    sys_prompt=default_system_prompt
    usr_prompt=description_text
    return sys_prompt,usr_prompt

model_name="gpt_4o"
out_filename=PROJDIR / "data/3_processed/dataset_2310/downstream" / "eval_data" /("aud_res_abs_k1_"+model_name+".jsonl")
make_batch(dict_df_with_dummy,out_filename,model_name=model_name,prompt_gen_func=default_prompt_dummy_k1)


# %%k4
DOWNSTREAM_PATH = PROJDIR/"data/3_processed/dataset_2310/downstream"
filename_openai_rst = DOWNSTREAM_PATH / "eval_data/prep_eval_abs_k4_gpt_4o_output.jsonl"
ans_list = get_results_openai_batch(filename_openai_rst,json=False)
ans_df = pd.DataFrame(ans_list)
ans_df.index_num = ans_df.index_num.str.replace("request-","")#.astype(int)
dict_df_with_dummy = pd.merge(dict_df,ans_df.rename(columns={'output':'dummy_description'}),left_on='index_num',right_on='index_num',how='left')
dict_df_with_dummy.dummy_description = dict_df_with_dummy.dummy_description.apply(delete_title)

model_name="gpt_4o"
out_filename=PROJDIR / "data/3_processed/dataset_2310/downstream" / "eval_data" /("aud_res_abs_k4_"+model_name+".jsonl")
make_batch(dict_df_with_dummy,out_filename,model_name=model_name,prompt_gen_func=default_prompt_dummy_k1)

# %%

# %% (eval 関連性) noise added answer generation

# %%

import random
random.seed(0)
    
def validate_eval_adddummy_far_prompt(sr:pd.Series):
    """
        3(eval 関連性) halsination invoked
        論点が似ている別の会社の検討事項ように書き換えてもらう
        別の手続をn個追加する
    """

    instruction_1="監査担当者であるあなたは、次の監査上の検討事項を与えられました。"
    instruction_2="これに対応する監査上の対応事項を日本語文章で具体的に立案してください。"

    description_text=sr.description
    c_id=sr.id
    tar_vec=GET_VECTOR_VALID_OBJ.get_vecter(c_id)
    nearest_idx=GET_VECTOR_VALID_OBJ.get_farthest_id([tar_vec],k=1)
    proc_list = GET_VECTOR_VALID_OBJ.proc_instance(nearest_idx[0][0])

    #r=sr.r
    proc_list=[list(item_dict.values())[0] for item_dict in proc_list]
    s=1#max(round(len(proc_list)*r),1)
    proc_smp_list=random.sample(proc_list,s)
    hint_text="#### ヒント\n次の対応事項を含めてください。\n * "+"\n * ".join(proc_smp_list)
    sys_prompt=instruction_1+"\n\n#### 検討事項\n"+description_text
    usr_prompt=instruction_2+"\n\n"+hint_text+"\n\n#### 監査上の対応事項"
    return sys_prompt,usr_prompt

def validate_eval_adddummy_far_prompt_all(sr:pd.Series):
    """
        3(eval 関連性) halsination invoked
        論点が似ている別の会社の検討事項ように書き換えてもらう
        別の手続をn個追加する
    """

    instruction_1="監査担当者であるあなたは、次の監査上の検討事項を与えられました。"
    instruction_2="これに対応する監査上の対応事項を日本語文章で具体的に立案してください。"

    description_text=sr.description
    c_id=sr.id
    tar_vec = GET_VECTOR_VALID_OBJ.get_vecter(c_id)
    nearest_idx = GET_VECTOR_VALID_OBJ.get_farthest_id([tar_vec],k=1)
    proc_list = GET_VECTOR_VALID_OBJ.proc_instance(nearest_idx[0][0])

    #r=sr.r
    proc_list=[list(item_dict.values())[0] for item_dict in proc_list]
    #proc_smp_list=random.sample(proc_list,s)
    hint_text="#### ヒント\n次の対応事項を含めてください。\n * "+"\n * ".join(proc_list)
    sys_prompt=instruction_1+"\n\n#### 検討事項\n"+description_text
    usr_prompt=instruction_2+"\n\n"+hint_text+"\n\n#### 監査上の対応事項"
    return sys_prompt,usr_prompt



GET_VECTOR_VALID_OBJ=get_validation_data_vector()
filename=PROJDIR / "data/3_processed/dataset_2310/downstream" / "2_intermediate/llm_proc" /"audit_res_markdown_eval.csv"
dict_df=pd.read_csv(filename,index_col=None,dtype=str).set_index('index_num')


model_name="gpt_4o"
out_filename=PROJDIR / "data/3_processed/dataset_2310/downstream" / "eval_data" /("aud_res_eval_rel_"+model_name+"_add_ans_far.jsonl")
make_batch(dict_df,out_filename,model_name=model_name,prompt_gen_func=validate_eval_adddummy_far_prompt)

model_name="gpt_4o"
out_filename=PROJDIR / "data/3_processed/dataset_2310/downstream" / "eval_data" /("aud_res_eval_rel_"+model_name+"_add_ans_far_all.jsonl")
make_batch(dict_df,out_filename,model_name=model_name,prompt_gen_func=validate_eval_adddummy_far_prompt_all)



# %% QAG -> next study
# 改善点の指摘
# 網羅性の指摘 1つ欠損させて手続を与えて、予測してもらう











# %% #########################################################
# memo
##############################################################
default_system_prompt="監査担当者であるあなたは、次の監査上の検討事項を与えられました。これに対応する監査上の対応事項を日本語文章で具体的に立案してください。"
inf_list=[]
for itr_index_num in dict_df.head().index:
    description_text=dict_df.loc[itr_index_num,'description']
    system_prompt=default_system_prompt
    usr_prompt=description_text
    messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": usr_prompt},
        ]
    #"gpt-3.5-turbo-0125"
    #"gpt-4o-2024-08-06"
    model_name="gpt-4o-mini"
    temp={
        "custom_id": "request-"+str(itr_index_num),
        "method": "POST", "url": "/v1/chat/completions",
        "body": {
            "model": model_name,
            "messages":messages,
            "temperature": 1,
            "max_tokens": 1024
            }}
    inf_list.append(temp)

out_filename="/Users/noro/Documents/Projects/XBRL_common_space_projection/tests/20241014/test.jsonl"
with open(out_filename, 'w') as file:
    for obj in inf_list:
        file.write(json.dumps(obj) + '\n')

batch_input_file = openai_api_obj.client.files.create(
  file=open(out_filename, "rb"),
  purpose="batch"
)
print(batch_input_file)
#
batch_job = openai_api_obj.client.batches.create(
    input_file_id=batch_input_file.id,
    endpoint="/v1/chat/completions",
    completion_window="24h",
    metadata={
      "description": "test-監査"
    }
)
#
batch_output = pd.read_json('/Users/noro/Documents/Projects/XBRL_common_space_projection/tests/20241014/batch_670d255f9660819081e9c30f9f7113e7_output.jsonl', orient='records', lines=True)
batch_output.response[0]['body']['choices'][0]['message']['content']
# batch make dataset for eval
sys_prompt, usr_prompt = make_prompt_qag(prompt_dict,ans_text)

inf_list=[]
for itr_index_num in dict_df.head().index:
    description_text=dict_df.loc[itr_index_num,'description']
    system_prompt=default_system_prompt
    usr_prompt=description_text
    messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": usr_prompt},
        ]
    #"gpt-3.5-turbo-0125"
    #"gpt-4o-2024-08-06"
    model_name="gpt-4o-mini"
    temp={
        "custom_id": "request-"+str(itr_index_num),
        "method": "POST", "url": "/v1/chat/completions",
        "body": {
            "model": model_name,
            "messages":messages,
            "temperature": 1,
            "max_tokens": 1024
            }}
    inf_list.append(temp)

out_filename="/Users/noro/Documents/Projects/XBRL_common_space_projection/tests/20241014/test.jsonl"
with open(out_filename, 'w') as file:
    for obj in inf_list:
        file.write(json.dumps(obj) + '\n')

filename="/Users/noro/Documents/Projects/XBRL_common_space_projection/data/3_processed/dataset_2310/downstream/0831/eval/batch_670da85949c8819080c6aa206c813a82_output.jsonl"
batch_output = pd.read_json(filename, orient='records', lines=True)
text=batch_output.response[0]['body']['choices'][0]['message']['content']
print(text)




# %% memo #########################################################


# %% old version


def validate_eval_abstract_prompt_far(sr:pd.Series):
    instruction_1="監査担当者であるあなたは、監査上の検討事項を与えられたら、対応する監査上の対応事項を立案します。"
    instruction_2="次の検討事項が与えられました。\n\n#### 検討事項\n"
    #instruction_3="\nこれに対応する監査上の対応事項は次のように立案されます。\n\n####監査上の対応事項\n"
    
    #instruction_4="次の検討事項が与えられました。\n\n#### 検討事項\n"
    #instruction_5="\nこれに対応する監査上の対応事項は次のように立案されます。\n\n####監査上の対応事項\n"
    
    instruction_6="\nこのような検討事項について、一般にどのような監査上の対応事項が必要か日本語文章で具体的に立案してください。"
    c_id=sr.id
    description_text=sr.description
    ans_text=sr.audit_res
    tar_vec=GET_VECTOR_VALID_OBJ.get_vecter(c_id)
    nearest_idx=GET_VECTOR_VALID_OBJ.get_nearest_id(tar_vec,k=300)
    shot_description_text, shot_audit_res = GET_VECTOR_VALID_OBJ.validation_instance(nearest_idx[300-1][0])
    sys_prompt=instruction_1
    usr_prompt=instruction_2+description_text+"\n\n"+shot_description_text+instruction_6

    return sys_prompt,usr_prompt

GET_VECTOR_VALID_OBJ=get_validation_data_vector()

model_name="gpt_4o_mini"
out_filename=PROJDIR / "data/3_processed/dataset_2310/downstream" / "baseline" /("batch_gen_audres_eval_absfar_"+model_name+".jsonl")
make_batch(dict_df,out_filename,model_name=model_name,prompt_gen_func=validate_eval_abstract_prompt_far)



def validate_eval_abstract_prompt_old(sr:pd.Series):
    instruction_1="監査担当者であるあなたは、監査上の検討事項を与えられたら、対応する監査上の対応事項を立案します。"
    instruction_2="例えば、次の検討事項が与えられました。\n\n#### 検討事項\n"
    instruction_3="\nこれに対応する監査上の対応事項は次のように立案されます。\n\n####監査上の対応事項\n"
    
    instruction_4="例えば、次の検討事項が与えられました。\n\n#### 検討事項\n"
    instruction_5="\nこれに対応する監査上の対応事項は次のように立案されます。\n\n####監査上の対応事項\n"
    
    instruction_6="\n以上をまとめると、このような検討事項について、一般にどのような監査上の対応事項が必要か日本語文章で具体的に立案してください。"
    c_id=sr.id
    description_text=sr.description
    ans_text=sr.audit_res
    tar_vec=GET_VECTOR_VALID_OBJ.get_vecter(c_id)
    nearest_idx=GET_VECTOR_VALID_OBJ.get_nearest_id(tar_vec)
    shot_description_text, shot_audit_res = GET_VECTOR_VALID_OBJ.validation_instance(nearest_idx[0][0])
    sys_prompt=instruction_1
    usr_prompt=instruction_2+description_text+instruction_3+ans_text+instruction_4+shot_description_text+instruction_5+shot_audit_res+instruction_6

    return sys_prompt,usr_prompt

GET_VECTOR_VALID_OBJ=get_validation_data_vector()

model_name="gpt_4o_mini"
out_filename=PROJDIR / "data/3_processed/dataset_2310/downstream" / "baseline" /("batch_gen_audres_eval_abs_"+model_name+".jsonl")
make_batch(dict_df,out_filename,model_name=model_name,prompt_gen_func=validate_eval_abstract_prompt_old)


def validate_eval_abstract_prompt_far_old(sr:pd.Series):
    instruction_1="監査担当者であるあなたは、監査上の検討事項を与えられたら、対応する監査上の対応事項を立案します。"
    instruction_2="例えば、次の検討事項が与えられました。\n\n#### 検討事項\n"
    instruction_3="\nこれに対応する監査上の対応事項は次のように立案されます。\n\n####監査上の対応事項\n"
    
    instruction_4="例えば、次の検討事項が与えられました。\n\n#### 検討事項\n"
    instruction_5="\nこれに対応する監査上の対応事項は次のように立案されます。\n\n####監査上の対応事項\n"
    
    instruction_6="\n以上をまとめると、このような検討事項について、一般にどのような監査上の対応事項が必要か日本語文章で具体的に立案してください。"
    c_id=sr.id
    description_text=sr.description
    ans_text=sr.audit_res
    tar_vec=GET_VECTOR_VALID_OBJ.get_vecter(c_id)
    nearest_idx=GET_VECTOR_VALID_OBJ.get_nearest_id(tar_vec,k=500)
    shot_description_text, shot_audit_res = GET_VECTOR_VALID_OBJ.validation_instance(nearest_idx[500-1][0])
    sys_prompt=instruction_1
    usr_prompt=instruction_2+description_text+instruction_3+ans_text+instruction_4+shot_description_text+instruction_5+shot_audit_res+instruction_6

    return sys_prompt,usr_prompt

GET_VECTOR_VALID_OBJ=get_validation_data_vector()

model_name="gpt_4o_mini"
out_filename=PROJDIR / "data/3_processed/dataset_2310/downstream" / "baseline" /("batch_gen_audres_eval_absfar_"+model_name+".jsonl")
make_batch(dict_df,out_filename,model_name=model_name,prompt_gen_func=validate_eval_abstract_prompt_far_old)

