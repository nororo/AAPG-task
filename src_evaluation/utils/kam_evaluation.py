

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
import re


# %%
PROJPATH=r"PROJECT_PATH"
PROJDIR=Path(PROJPATH)



class eval_output():
    def __init__(self, prompt_path:str,out_dir:str,model_name_extract:str="gpt_4o_mini",model_name_judge:str="gpt_4o_mini",model_name_judge_rel:str="gpt_4o",trial_flg:bool=True):
        with open(prompt_path, mode="rt", encoding="utf-8") as f:
        	prompt_dict = json.load(f)    
        self.prompt_dict=prompt_dict
        self.out_dir=Path(out_dir)
        self.eval_rst_template={
            'comp_llm':None,
            'conc_llm':None,
            'rel_llm':None,
            'cost_dollar':0
            }
        self.model_name_extract=model_name_extract
        self.model_name_judge=model_name_judge
        self.model_name_judge_rel=model_name_judge_rel
        
        filename_openai_rst=PROJDIR / "data/3_processed/dataset_2310/downstream" / "eval_prep" /"eval_mask_qa_from_process_gpt4_output.jsonl"
        response_list=get_results_openai_batch(filename_openai_rst=filename_openai_rst)
        response_list_df=pd.DataFrame(response_list)
        response_list_df=response_list_df.assign(
            index_num_df=response_list_df.index_num.str.replace("request-","").str.split("_",expand=True)[0],
            index_num_qa=response_list_df.index_num.str.replace("request-","").str.split("_",expand=True)[1]
            )
        self.mask_qa_list=response_list_df.query("status!='Failed'")
        
        # validation data
        filename=PROJDIR / "data/3_processed/dataset_2310/downstream" / "2_intermediate/llm_proc" /"audit_res_markdown_eval.csv"
        if trial_flg:
            self.data_val_df=pd.read_csv(filename,index_col=None,dtype=str).set_index('index_num').head(100)
        else:
            self.data_val_df=pd.read_csv(filename,index_col=None,dtype=str).set_index('index_num')
        
        # load extracted process
        filename_openai_rst_proc=PROJDIR / "data/3_processed/dataset_2310/downstream" / "eval_prep" /"eval_extracted_process_from_ans_gpt4_output.jsonl"
        response_list_df=pd.DataFrame(get_results_openai_batch(filename_openai_rst=filename_openai_rst_proc)).query("status!='Failed'")
        response_list_df=response_list_df.assign(
            index_num_df=response_list_df.index_num.str.replace("request-","")
            )
        response_list_df = response_list_df.set_index('index_num_df').rename(
            columns={'status':'prep_status','output':'prep_output'}
            )
        #self.data_val_df_con=pd.concat([self.data_val_df,response_list_df],axis=1)
        self.data_val_df_con=pd.merge(self.data_val_df,response_list_df,left_index=True,right_index=True,how='left')

        # load keywords        
        filename_openai_keyword=PROJDIR / "data/3_processed/dataset_2310/downstream" / "eval_prep" /"eval_keywords_from_ans_gpt4_output.jsonl"
        response_list_df = pd.DataFrame(get_results_openai_batch(filename_openai_rst=filename_openai_keyword)).query("status!='Failed'")
        response_list_df=response_list_df.assign(
            index_num_df=response_list_df.index_num.str.replace("request-","")
            )
        self.keywords_df = response_list_df.set_index('index_num_df')#.rename(columns={'status':'keyword_status','output':'keyword_output'}) 
        self.completeness_mask_qag_ans_list=[]
        self.concreteness_score=[]

    def gen_eval_batch(self,proposed_res:list,out_filename_without_ext):
        # set up batch_inf_file_generator
        self.batch_inf_file_generator_obj=batch_inf_file_generator(prompt_dict=self.prompt_dict)
        self.batch_inf_file_generator_obj_4o=batch_inf_file_generator(prompt_dict=self.prompt_dict)
        
        for index_num_str in self.data_val_df.index:
            self.completeness_mask_qag(proposed_res[int(index_num_str)],index_num_str)
            self.completeness_llm(proposed_res[int(index_num_str)],index_num_str)
            self.concreteness_llm(proposed_res[int(index_num_str)],index_num_str)
        self.export_batch_file_openai(out_filename_without_ext)
        
        out_filename=self.out_dir / (out_filename_without_ext+"mask_qag_ans.csv")
        pd.DataFrame(self.completeness_mask_qag_ans_list).to_csv(out_filename,index=False)

        out_filename=self.out_dir / (out_filename_without_ext+"conc_keyword_score.csv")
        pd.DataFrame(self.concreteness_score).to_csv(out_filename,index=False)

    def gen_eval_batch_rel(self,proposed_res:list,out_filename_without_ext):
        # set up batch_inf_file_generator
        self.batch_inf_file_generator_obj_4o_rel=batch_inf_file_generator(prompt_dict=self.prompt_dict)
        
        for index_num_str in self.data_val_df.index:
            self.relevancy_llm(proposed_res[int(index_num_str)],index_num_str)
        
    def gen_and_expt_eval_batch_hal(self,proposed_res:list,out_filename_without_ext):
        # set up batch_inf_file_generator
        self.batch_inf_file_generator_obj_hal=batch_inf_file_generator(prompt_dict=self.prompt_dict)
        
        for index_num_str in self.data_val_df.index:
            self.faithfulness_hallucination_llm(proposed_res[int(index_num_str)],index_num_str)
        out_filename_openai=self.out_dir / (out_filename_without_ext+"_4o_test_hal.jsonl")
        self.batch_inf_file_generator_obj_hal.export_list(out_filename_openai)
        


    def export_batch_file_openai(self,out_filename_without_ext):
        out_filename_openai=self.out_dir / (out_filename_without_ext+"test.jsonl")
        self.batch_inf_file_generator_obj.export_list(out_filename_openai)
        out_filename_openai=self.out_dir / (out_filename_without_ext+"_4o_test.jsonl")
        self.batch_inf_file_generator_obj_4o.export_list(out_filename_openai)
        out_filename_openai=self.out_dir / (out_filename_without_ext+"_4o_test_rel.jsonl")
        self.batch_inf_file_generator_obj_4o_rel.export_list(out_filename_openai)
        

    def completeness_mask_qag(self,proposed_res,index_num_str:str):
        df_qa=self.mask_qa_list.query("index_num_df==@index_num_str")
        itr_index_num=0
        
        ans_list=[]
        for index_num_qa in df_qa.index_num_qa:
            qa_list=df_qa.query("index_num_qa==@index_num_qa").iloc[0,:].output
            for qa_num,qa in enumerate(qa_list):
                q_text=qa['置換後の文章']
                system_prompt_comp=self.prompt_dict['qag_instruction']+"\n\n#### 問題文\n"+q_text
                user_prompt_comp="#### 提供された文章\n"+proposed_res+"\n"+self.prompt_dict['qag_output_format']
                itr_index_str="maskqag"+"-"+index_num_str+"_"+index_num_qa+"_"+str(qa_num) # reqest-${evalname}-{${num}_${num}_${num}}
                self.batch_inf_file_generator_obj.insert_inf_list_prompt(
                    system_prompt_comp,user_prompt_comp,itr_index_str,max_tokens=128,model_name=self.model_name_extract
                    )
                a_text=qa['置換した用語']
                ans_list.append({"index_num":"request-"+itr_index_str,"ans":a_text})
                itr_index_num=itr_index_num+1
        #print("===== completeness_mask_qag =====")
        self.completeness_mask_qag_ans_list=self.completeness_mask_qag_ans_list+ans_list
        
    def completeness_llm(self,proposed_res,index_num_str,version_num=2):
        ans_text=self.data_val_df.loc[index_num_str,'output']
        if version_num==1:
            sys_prompt, usr_prompt = make_prompt_eval_comp(self.prompt_dict,ans_text,proposed_res)
        elif version_num==2:
            sys_prompt, usr_prompt = make_prompt_eval_comp2(self.prompt_dict,ans_text,proposed_res)
        itr_index_str="llmcomp"+"-"+index_num_str # reqest-${evalname}-{${num}}
        self.batch_inf_file_generator_obj_4o.insert_inf_list_prompt(sys_prompt,usr_prompt,itr_index_str,max_tokens=1024,model_name=self.model_name_judge)
        #print("===== completeness_llm =====")
        
    
    def concreteness_llm(self,proposed_res,index_num):
        ans_text=self.data_val_df.loc[index_num,'output']
        description_text=self.data_val_df.loc[index_num,'description']
        sys_prompt, usr_prompt = make_prompt_eval_conc(self.prompt_dict,description_text,ans_text,proposed_res)
        itr_index_str="llmcomc"+"-"+index_num # reqest-${evalname}-{${num}}
        self.batch_inf_file_generator_obj_4o.insert_inf_list_prompt(sys_prompt,usr_prompt,itr_index_str,max_tokens=1024,model_name=self.model_name_judge)
        #print("===== concreteness_llm =====")
        
    def relevancy_llm(self,proposed_res,index_num_str):
        ans_text=self.data_val_df.loc[index_num_str,'output']
        description_text=self.data_val_df.loc[index_num_str,'description']
        sys_prompt, usr_prompt = make_prompt_eval_rel(self.prompt_dict,description_text,ans_text,proposed_res)
        itr_index_str="llmrel"+"-"+index_num_str # reqest-${evalname}-{${num}}
        self.batch_inf_file_generator_obj_4o_rel.insert_inf_list_prompt(sys_prompt,usr_prompt,itr_index_str,max_tokens=1024,model_name=self.model_name_judge_rel)
        #print("===== relevancy_llm =====")
        
    def faithfulness_hallucination_llm(self,proposed_res,index_num_str):
        ans_text=self.data_val_df.loc[index_num_str,'output']
        description_text=self.data_val_df.loc[index_num_str,'description']
        sys_prompt, usr_prompt = make_prompt_eval_hal(self.prompt_dict,description_text,ans_text,proposed_res)
        itr_index_str="llmhal"+"-"+index_num_str # reqest-${evalname}-{${num}}
        self.batch_inf_file_generator_obj_hal.insert_inf_list_prompt(sys_prompt,usr_prompt,itr_index_str,max_tokens=1024,model_name=self.model_name_judge)
        

def eval_output_old(prompt_dict,openai_api_obj,description,ans_text,output_text):
    eval_rst_t={
        'comp_llm':None,
        'conc_llm':None,
        'rel_llm':None,
        'cost_dollar':0
    }

    print("completeness")
    sys_prompt, usr_prompt = make_prompt_eval_comp(prompt_dict,ans_text,output_text)
    output=openai_api_obj.request_rapper(
        system_prompt=sys_prompt,
        usr_prompt=usr_prompt,
        model_name='gpt_4o_mini',
        max_tokens=128)
    
    score=extract_json_dict(output["output"],required_output_num=1)
    eval_rst_t['comp_llm']=re.sub(r"\D","",str(score))
    eval_rst_t['comp_llm_output']=output["output"]
    eval_rst_t['cost_dollar']=eval_rst_t['cost_dollar']+output['cost_dollar']
    # concreteness
    print("concreteness")
    sys_prompt, usr_prompt = make_prompt_eval_conc(prompt_dict,description,ans_text,output_text)
    output=openai_api_obj.request_rapper(
        system_prompt=sys_prompt,
        usr_prompt=usr_prompt,
        model_name='gpt_4o_mini',
        max_tokens=128)

    score=extract_json_dict(output["output"],required_output_num=1)
    eval_rst_t['conc_llm']=re.sub(r"\D","",str(score))
    eval_rst_t['conc_llm_output']=output["output"]
    eval_rst_t['cost_dollar']=eval_rst_t['cost_dollar']+output['cost_dollar']

    # relevancy
    print("relevancy")
    sys_prompt, usr_prompt = make_prompt_eval_rel(prompt_dict,description,ans_text,output_text)
    output=openai_api_obj.request_rapper(
        system_prompt=sys_prompt,
        usr_prompt=usr_prompt,
        model_name='gpt_4o_mini',
        max_tokens=128)
    
    score=extract_json_dict(output["output"],required_output_num=1)
    eval_rst_t['rel_llm']=re.sub(r"\D","",str(score))
    eval_rst_t['rel_llm_output']=output["output"]
    eval_rst_t['cost_dollar']=eval_rst_t['cost_dollar']+output['cost_dollar']

    return eval_rst_t
