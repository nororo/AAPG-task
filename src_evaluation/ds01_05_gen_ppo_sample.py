
"""
preparation for evaluation

"""
# %%
from libs.model_api import *
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

# %%
#from openai import OpenAI
load_dotenv(verbose=True)
dotenv_path = join(Path(dirname(__file__)).parents[1] / "env" / "k", '.env')
load_dotenv(dotenv_path)

# %%

######################################################
#
#            Prompt
#
######################################################
from libs.compose_prompt import *
from libs.utils import *


# %% step 1
# イレギュラー反映前に、こちらをベースに計算 (-> trial_llama3_1_convert_to_markdown.ipynb -> audit_res_markdown and audit_res_markdown_2)
filename=PROJDIR / "data/3_processed/dataset_2310/downstream/2_intermediate" / "data_all_pivot_0831.csv"
data_all_pivot_0831=pd.read_csv(filename)
# イレギュラー反映後のsim除外pool選択
filename=PROJDIR / "data/3_processed/dataset_2310/downstream/2_intermediate" / "data_all_pivot_1012.csv"
data_all_pivot_0831_v2=pd.read_csv(filename)
drop_set=set(data_all_pivot_0831.id)-set(data_all_pivot_0831_v2.id)



filename=PROJDIR / "data/3_processed/dataset_2310/downstream/2_intermediate/llm_proc" / "audit_res_markdown.csv"
audit_res_markdown=pd.read_csv(filename)
filename=PROJDIR / "data/3_processed/dataset_2310/downstream/2_intermediate/llm_proc" / "audit_res_markdown_2.csv"
audit_res_markdown_2=pd.read_csv(filename)
audit_res_markdown=pd.concat([audit_res_markdown.query("index_num<5151"),audit_res_markdown_2]).reset_index(drop=True)

# markdown converted (additional data(contains irr corrected and additional data))
filename=PROJDIR / "data/3_processed/dataset_2310/downstream/2_intermediate/llm_proc" / "audit_res_markdown_irr.csv"
audit_res_markdown_irr=pd.read_csv(filename)

# correction (drop irr missed data and add additional data)
dict_df=pd.concat([audit_res_markdown.query("id != 'S100NSFT_FilingDateInstant_Row1Member' and id not in @drop_set"),audit_res_markdown_irr],axis=0)
dict_df=dict_df.set_index('id')

# %%
prompt_qag_1 = {
    "qag_instruction": """提供された文章から、監査上の対応事項をできるだけ具体的に抽出してください。""",
    #### 注意事項
    "qag_constraints": ["抽出した監査上の対応事項には、提供された文章から省略された情報があれば補ってください。"],
    "qag_output_formats": """#### 回答形式\n\nフォーマットは個別のjson形式で回答してください。\n\n{"監査手続":"(監査手続1)"}\n{"監査手続":"(監査手続2)"}""",
    #### 文章
    # ${}
    }
#tmp=get_example_prompt(prompt_qag_1)

vertex_ai_api_obj = vertex_ai_api()
groq_api_obj = groq_api(gc_client=vertex_ai_api_obj)


def inf_recur(output_text,sys_prompt, usr_prompt, groq_api_obj,model_name,cnt=0):
    if cnt>5:
        return output_text, 'Failed'
    try:
        out_json_list=extract_json_dict(output_text)
        return output_text, 'Success'
    except Exception as e:
        print(e)
        output=groq_api_obj.request_rapper(
            system_prompt=sys_prompt,
            usr_prompt=usr_prompt,
            model_name=model_name,
            max_tokens=1024
            )
        cnt=cnt+1
        return inf_recur(output["output"],sys_prompt, usr_prompt, groq_api_obj,model_name,cnt)
        

# step 1
def main_1():
    out_filename=PROJDIR / "data/3_processed/dataset_2310/downstream" / "eval_prep" /"ppo_extracted_process_from_ans_1112.csv"
    last_hourly_task = time.time()
    with open(out_filename, mode='w', encoding='utf-8', newline='') as csv_file:
        fieldnames = [
            'index_num',
            'proc_text',
            'output',
            'api_status',
            'pred',
            'status'
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for index_num in tqdm(dict_df.index):
            current_time = time.time()
            if (current_time - last_hourly_task >= 2400):
                vertex_ai_api_obj = vertex_ai_api()
                groq_api_obj = groq_api(gc_client=vertex_ai_api_obj)
                last_hourly_task = current_time
            ans_text=dict_df.loc[index_num,'output']
            #sys_prompt, usr_prompt = make_prompt_qag(prompt_qag,ans_text)
            sys_cls_prompt, usr_cls_prompt = make_prompt_qag_prep(prompt_qag_1,ans_text)
            output=groq_api_obj.request_rapper(
                system_prompt=sys_cls_prompt,
                usr_prompt=usr_cls_prompt,
                model_name="llama_3.1_8b",
                max_tokens=1024
                )
            ans_t={
                    'index_num':index_num,
                    'proc_text':ans_text,
                    'output':'-',
                    'api_status':'-',
                    'pred':'-',
                    'status':'-',
                    }
            ans_t["api_status"]=output["status"]
            ans_t["output"],ans_t["status"]=inf_recur(output["output"],sys_cls_prompt, usr_cls_prompt, groq_api_obj,model_name="llama_3.1_8b")
            writer.writerow(ans_t)


# %%
def get_results_csv(filename_openai_rst):
    output = pd.read_csv(filename_openai_rst).set_index('index_num')
    ans_list=[]
    for itr_index in output.index:
        
        output_sr=output.loc[itr_index,:]
        ans_t={
                'index_num':str(itr_index),
                'output':'-',
                'api_status':'-',
                'pred':'-',
                'status':'-',
                }
        #output=output.response[itr_index]['body']['choices'][0]['message']['content']
        ans_t["output"]=output_sr["output"]
        ans_t["api_status"]=output_sr["api_status"]
        try:
            out_json_list=extract_json_dict(output_sr['output'])
            ans_t["output"]=out_json_list
            #for out_proc in out_json_list:
            #    proc_text=list(out_proc.values())[0]
            #    make_prompt_qag_prep(prompt_qag_2,proc_text)
            ans_t['status']='Success'
        except Exception as e:
            print(e)
            ans_t['status']='Failed'
        ans_list.append(ans_t)
    return ans_list


out_filename=PROJDIR / "data/3_processed/dataset_2310/downstream" / "eval_prep" /"ppo_extracted_process_from_ans_1112.csv"
response_list=get_results_csv(out_filename)
#
# %% step 2

# %% step2
prompt_qag_2 = {
    "qag_instruction": """提供された文章の匿名性を高めるため、特徴的な専門用語を1つ選択し、<MASK>に置換したアウトプット文章を2通り提供してください。""",
    #### 注意事項
    "qag_constraints": [
        "置換する専門用語は、一体で意味を成すひとつながりの用語を分けずに選択してください。例: 連結貸借対照表を選択する場合は「連結」や「貸借対照表」ではなく「連結貸借対照表」を選択します。",
        "置換する専門用語は1つのアウトプット文章につき1つとします。",
        "置換する専門用語は、それを置換したアウトプット文章のみからは推定できないような用語を選択してください。",
        "置換する専門用語は文章中の理由に関する記載範囲以外から選択してください。"
        ],
    "qag_output_formats": """#### 回答形式\n\nフォーマットは個別のjson形式で回答してください。\n\n{"置換後の文章":"(置換後の文章1)","置換した用語":"(置換した用語1)"}\n{"置換後の文章":"(置換後の文章2)","置換した用語":"(置換した用語2)"}""",
    #### 文章
    # ${}
    }
# %%
vertex_ai_api_obj = vertex_ai_api()
groq_api_obj = groq_api(gc_client=vertex_ai_api_obj)
out_filename=PROJDIR / "data/3_processed/dataset_2310/downstream" / "eval_prep" /"ppo_mask_qa_from_process_1114.csv"
last_hourly_task = time.time()
with open(out_filename, mode='w', encoding='utf-8', newline='') as csv_file:
    fieldnames = [
        'index_num',
        'id',
        'output',
        'api_status',
        'pred',
        'status'
    ]
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    index_num=0
    for itr,response in tqdm(enumerate(response_list)):
        
        id_str=response['index_num']
        itr_res=0
        for out_proc in response['output']:
            current_time = time.time()
            if (current_time - last_hourly_task >= 2400):
                vertex_ai_api_obj = vertex_ai_api()
                groq_api_obj = groq_api(gc_client=vertex_ai_api_obj)
                last_hourly_task = current_time
            
            #try:
            proc_text=list(out_proc.values())[0]
            if type(proc_text)==list:
                for itr_proc_text in proc_text:
                    index_str=str(id_str)+'_'+str(itr_res)
                    ans_t={
                        'index_num':index_str,
                        'id':id_str,
                        'output':'-',
                        'api_status':'-',
                        'pred':'-',
                        'status':'-',
                        }
                    sys_cls_prompt, usr_cls_prompt = make_prompt_qag_prep(prompt_qag_2,str(itr_proc_text))
                    output=groq_api_obj.request_rapper(
                        system_prompt=sys_cls_prompt,
                        usr_prompt=usr_cls_prompt,
                        model_name="llama_3.1_70b",
                        max_tokens=1024
                        )
                    ans_t["api_status"]=output["status"]
                    ans_t["output"],ans_t["status"]=inf_recur(output["output"],sys_cls_prompt, usr_cls_prompt, groq_api_obj,model_name="llama_3.1_70b")
                    itr_res=itr_res+1
                    writer.writerow(ans_t)

            elif type(proc_text)==str:
                index_str=str(id_str)+'_'+str(itr_res)
                ans_t={
                    'index_num':index_str,
                    'id':id_str,
                    'output':'-',
                    'api_status':'-',
                    'pred':'-',
                    'status':'-',
                    }

                sys_cls_prompt, usr_cls_prompt = make_prompt_qag_prep(prompt_qag_2,str(proc_text))
                output=groq_api_obj.request_rapper(
                    system_prompt=sys_cls_prompt,
                    usr_prompt=usr_cls_prompt,
                    model_name="llama_3.1_70b",
                    max_tokens=1024
                    )
                ans_t["api_status"]=output["status"]
                ans_t["output"],ans_t["status"]=inf_recur(output["output"],sys_cls_prompt, usr_cls_prompt, groq_api_obj,model_name="llama_3.1_70b")
                itr_res=itr_res+1
            #except Exception as e:
            #    print(e)
            #    ans_t['status']='Failed'
                writer.writerow(ans_t)



# %%

filename="/Users/noro/Documents/Projects/XBRL_common_space_projection/data/3_processed/dataset_2310/downstream/ppo_sample/1114/ppo_mask_qa_from_process_1114.csv"
ppo_sample=pd.read_csv(filename)
ppo_sample=ppo_sample.assign(
    index_num_instance=ppo_sample.index_num.str.split("_",expand=True)[[0,1,2]].apply(lambda x: "_".join(x),axis=1),
    index_num_proc=ppo_sample.index_num.str.split("_",expand=True)[3],
)

ppo_smp_list=[]
error_smp_list=[]
for itr_index in ppo_sample.index:
    try:
        qa_list=extract_json_dict(ppo_sample.loc[itr_index,'output'])
        for qa_num,qa in enumerate(qa_list):
            q=list(qa_list[0].values())[0]
            a=list(qa_list[0].values())[1]
            ppo_smp_temp={
                'id':str(ppo_sample.loc[itr_index,'index_num_instance']),
                'process_num_instance':int(ppo_sample.loc[itr_index,'index_num_proc']),
                'qa_num':str(qa_num),
                'mask_q':q,
                'mask_a':a,
                'len_mask_q':len(q),
                'len_mask_a':len(a),
            }
            ppo_smp_list.append(ppo_smp_temp)
    except Exception as e:
        error_smp_temp={
            'id':str(ppo_sample.loc[itr_index,'index_num_instance']),
            'process_num_instance':int(ppo_sample.loc[itr_index,'index_num_proc']),
            'output':ppo_sample.loc[itr_index,'output'],
            'message':str(e),
            }
        error_smp_list.append(error_smp_temp)
        print(e)

#filename="/Users/noro/Documents/Projects/XBRL_common_space_projection/data/3_processed/dataset_2310/downstream/ppo_sample/1114/ppo_extracted_process_from_ans_1112.csv"
#ppo_process=pd.read_csv(filename).rename(columns={'index_num':'id'})

ppo_smp_list_df=pd.DataFrame(ppo_smp_list)
ppo_smp_list_df=ppo_smp_list_df.query('len_mask_q+len_mask_a>26')
ppo_smp_list_df['cnt']=1
#ppo_smp_list_df=pd.merge(ppo_smp_list_df,ppo_process[['id']],left_on='index_num_instance',right_index=True,how='left')

ppo_smp_list_df_g=ppo_smp_list_df.groupby('id').agg({'cnt':'sum','id':'first'}).sort_values('cnt',ascending=False)
id_set=set(ppo_smp_list_df_g.query("cnt>=4").id)
ppo_smp_list_df_expt=ppo_smp_list_df.query("id in @id_set")
ppo_smp_list_df_expt.to_csv("/Users/noro/Documents/Projects/XBRL_common_space_projection/data/3_processed/dataset_2310/downstream/ppo_sample/1114/ppo_mask_qa_dataset_1114.csv",index=False)