
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

# %%
PROJPATH=r"PROJECT_PATH"
PROJDIR=Path(PROJPATH)

# %%
#from openai import OpenAI
load_dotenv(verbose=True)
dotenv_path = join(Path(dirname(__file__)).parents[1] / "env" / "k", '.env')
load_dotenv(dotenv_path)
openai_api_obj=openai_api()



# %% #########################################################
#                    RESULTS
##############################################################

from rouge_score.rouge_scorer import RougeScorer
from rouge_score.tokenizers import Tokenizer
from rouge_score.tokenize import SPACES_RE
class NonAlphaNumericSupportTokenizer(Tokenizer):

    def tokenize(self, text):
        return SPACES_RE.split(text.lower())


class calc_rouge():
    """
    https://nikkie-ftnext.hatenablog.com/entry/why-rouge-score-library-cannot-calculate-from-japanese-texts

    argument that is first:target next:pred follows;
        google-research/rouge/rouge_scorer.py at master · google-research/google-research
        https://github.com/google-research/google-research/blob/master/rouge/rouge_scorer.py
        def score(self, target, prediction):
            ...
    """
    def __init__(self):
        self.scorer = RougeScorer(
            ["rouge1", "rouge2", "rougeL", "rougeLsum"],
            tokenizer=NonAlphaNumericSupportTokenizer(),
        )    
    def calc_rouge(self,text_target:str,text_pred:str):
        tok_text_pred=tokenize_sudachi_for_series(text_pred)
        tok_text_target=tokenize_sudachi_for_series(text_target)    
        scores = self.scorer.score(" ".join(tok_text_target), " ".join(tok_text_pred))
        return scores['rougeL'].fmeasure
    
    def calc_rouge_precision(self,text_target:str,text_pred:str):
        tok_text_pred=tokenize_sudachi_for_series(text_pred)
        tok_text_target=tokenize_sudachi_for_series(text_target)    
        scores = self.scorer.score(" ".join(tok_text_target), " ".join(tok_text_pred))
        return scores['rougeL'].precision

    def eval_rouge(self,sr1,sr2):
        rouge_score_list=[]
        for x1,x2 in zip(sr1,sr2):
            rouge_score_list.append(self.calc_rouge(x1,x2))
        return rouge_score_list
        

def extract_ans(item,rtn='text'):
    """
    -> text(str) or score(float)
    """
    try:
        ans_item=list(extract_json_dict(item)[0].values())[0]
        if rtn=='text':
            return ans_item
        elif rtn=='score':
            return float(re.sub(r"\D","",str(ans_item)))
        #output_list.append(ans_item)
    except Exception as e:
        print(e)
        if rtn=='text':
            return '-'
        elif rtn=='score':
            return np.nan
        else:
            return None
# %% openai batch input 
def get_input_openai_batch(filename_openai_input):
    with open(filename_openai_input) as f:
        data = f.readlines()
        data = [json.loads(line) for line in data]
    return data

def get_fill():
    dirpath=Path("/Users/noro/Documents/Projects/XBRL_common_space_projection/data/3_processed/dataset_2310/downstream/error_retry_list")
    #self.dirpath = dirpath
    response_list_eval=[]
    #assert len(list(dirpath.glob("batch_*output.jsonl")))==2,"batch_*.jsonl files are {}".format(len(list(dirpath.glob("batch_*.jsonl"))))
    for file in dirpath.glob("batch_*output.jsonl"):
        response_list_eval=response_list_eval+get_results_openai_batch(filename_openai_rst=file,json=False)
    
    response_list_eval_df=pd.DataFrame(response_list_eval).query(
        "not index_num.str.contains('llmrel')"
        )
    # and not index_num.str.contains('llmcomc') and not index_num.str.contains('llmcomp')
    
    assert len(list(dirpath.glob("rel_batch_*output.jsonl")))==1,"rel_batch_*.jsonl files are {}".format(len(list(dirpath.glob("rel_batch_*.jsonl"))))
    
    for file in dirpath.glob("rel_batch_*output.jsonl"):
        response_list_eval_rel=get_results_openai_batch(filename_openai_rst=file,json=False)

    response_list_eval_rel_df = pd.DataFrame(response_list_eval_rel)
    response_list_eval_rel_df.index_num = response_list_eval_rel_df.index_num.str.split("__",expand=True)[0]
    response_list_eval_df=pd.concat([response_list_eval_df,response_list_eval_rel_df],axis=0)

    response_list_eval_df=response_list_eval_df.assign(
        eval_task=response_list_eval_df.index_num.str.split("-",expand=True)[1],
        num_eval_task=response_list_eval_df.index_num.str.split("-",expand=True)[2]
        )
    return response_list_eval_df
    #filename_mask_qag_ans=str(PROJDIR / "data/3_processed/dataset_2310/downstream/baseline" / "eval_plane_gpt_4o_minimask_qag_ans.csv")
    


# %%
class post_proc_eval():
    def __init__(self,filename_openai_rst,filename_mask_qag_ans,filename_conc_keyword_score,out_ext=""):
        dirpath=Path(str(filename_openai_rst))
        self.dirpath = dirpath
        response_list_eval=[]
        for file in dirpath.glob("batch_*output.jsonl"):
            response_list_eval=response_list_eval+get_results_openai_batch(filename_openai_rst=file,json=False)
        
        response_list_eval_df=pd.DataFrame(response_list_eval).query(
            "not index_num.str.contains('llmrel')"
            )
        
        assert len(list(dirpath.glob("rel_batch_*output.jsonl")))==1,"rel_batch_*.jsonl files are {}".format(len(list(dirpath.glob("rel_batch_*.jsonl"))))
        
        for file in dirpath.glob("rel_batch_*output.jsonl"):
            response_list_eval_rel=get_results_openai_batch(filename_openai_rst=file,json=False)

        response_list_eval_df=pd.concat([response_list_eval_df,pd.DataFrame(response_list_eval_rel)],axis=0)

        
        comp_flg=False
        if comp_flg:
            assert len(list(dirpath.glob("comp_batch_*output.jsonl")))==1,"comp_batch_*.jsonl files are {}".format(len(list(dirpath.glob("comp_batch_*.jsonl"))))

            for file in dirpath.glob("comp_batch_*output.jsonl"):
                response_list_eval_comp=get_results_openai_batch(filename_openai_rst=file,json=False)
            response_list_eval_df=pd.concat([response_list_eval_df,pd.DataFrame(response_list_eval_comp)],axis=0)
        

        response_list_eval_df=response_list_eval_df.assign(
            eval_task=response_list_eval_df.index_num.str.split("-",expand=True)[1],
            num_eval_task=response_list_eval_df.index_num.str.split("-",expand=True)[2]
            )
        self.response_list_eval_df=response_list_eval_df
        self.out_ext = out_ext
        
        self.mask_qag_ans=pd.read_csv(filename_mask_qag_ans)        

        self.filename_conc_keyword_score=filename_conc_keyword_score
        
        self.eval_results={
            "mask_qag":None,
            "comp_add_proc":None,
            "comp_llm":None,
            "conc_keyword_score":None,
            "conc_llm":None,
            "answer_relevancy":None,
            "relevancy_llm":None,
            "total":None
            }
        filename=PROJDIR/"eval_prep/nearest_score.csv"
        self.eval_results_df=pd.read_csv(filename)#.head(100)
        self.response_error_list = []
    def eval_and_export_results_hal(self):
        print("hal_llm")
        self.hal_llm()
        

    def eval_and_export_results(self):
        self.fill_df = get_fill()
        self.mask_qag()
        self.comp_llm()#
        self.conc_llm()#
        self.relevancy_llm()

        return self.eval_results

    def mask_qag(self):

        response_list_eval_df=self.response_list_eval_df
        mask_qag_df=response_list_eval_df.query("eval_task=='maskqag'")

        mask_qag_df=pd.merge(mask_qag_df,self.mask_qag_ans,left_on='index_num',right_on='index_num',how='left')
        mask_qag_df=mask_qag_df.assign(
            val_data_num=mask_qag_df.num_eval_task.str.split("_",expand=True)[0],
            proc_num=mask_qag_df.num_eval_task.str.split("_",expand=True)[1],
            sample_num=mask_qag_df.num_eval_task.str.split("_",expand=True)[2]
            )

        mask_qag_df['pred']=mask_qag_df.output.apply(extract_ans,args=('text',))
        
        fill_df_t = self.fill_df.query("index_num.str.contains('maskqag')").copy()
        fill_df_t['pred'] = fill_df_t.output.apply(extract_ans,args=('text',))
        mask = fill_df_t['pred']=='-'
        fill_df_t.loc[mask,'output']=fill_df_t.loc[mask,'output']+"""" }"""
        fill_df_t['pred']=fill_df_t.output.apply(extract_ans,args=('text',))

        fill_df_t = fill_df_t.set_index('index_num')
        
        mask_qag_df = mask_qag_df.set_index('index_num')
        mask_qag_df['pred'] = mask_qag_df['pred'].replace('-',None).fillna(fill_df_t.pred).fillna('-')
        mask_qag_df = mask_qag_df.reset_index()
        
        calc_rouge_obj=calc_rouge()
        mask_qag_df['rouge_f']=calc_rouge_obj.eval_rouge(mask_qag_df.pred.to_list(),mask_qag_df.ans.to_list())
        

        mask_qag_df_g=mask_qag_df.groupby('val_data_num').agg({'rouge_f':'mean'})
        self.eval_results["mask_qag"]=mask_qag_df_g.mean().values[0]
        self.eval_results_df["mask_qag"]=mask_qag_df_g.values
        self.response_error_list = self.response_error_list + mask_qag_df.query("pred=='-'").index_num.to_list()
    
    def comp_llm(self):
        llm_as_a_judge_df=self.response_list_eval_df.query("eval_task=='llmcomp'")
        llm_as_a_judge_df['score']=llm_as_a_judge_df.output.apply(extract_ans,args=('score',))
        
        fill_df_t = self.fill_df.query("not index_num.str.contains('maskqag')").copy()
        fill_df_t['score'] = fill_df_t.output.apply(extract_ans,args=('score',))
        fill_df_t = fill_df_t.set_index('index_num')
        llm_as_a_judge_df = llm_as_a_judge_df.set_index('index_num')
        llm_as_a_judge_df['score'] = llm_as_a_judge_df['score'].fillna(fill_df_t.score)
        llm_as_a_judge_df = llm_as_a_judge_df.reset_index()

        #self.llm_as_a_judge_comp_llm_df=llm_as_a_judge_df
        self.eval_results_df["comp_llm"]=llm_as_a_judge_df.score.values
        self.eval_results["comp_llm"]=llm_as_a_judge_df.score.mean()
        self.response_error_list = self.response_error_list + llm_as_a_judge_df.query("score.isna()").index_num.to_list()

    def relevancy_llm(self):
        llm_as_a_judge_df=self.response_list_eval_df.query("eval_task=='llmrel'")
        llm_as_a_judge_df['score']=llm_as_a_judge_df.output.apply(extract_ans,args=('score',))

        fill_df_t = self.fill_df.query("not index_num.str.contains('maskqag')").copy()
        fill_df_t['score'] = fill_df_t.output.apply(extract_ans,args=('score',))
        fill_df_t = fill_df_t.set_index('index_num')
        llm_as_a_judge_df = llm_as_a_judge_df.set_index('index_num')
        llm_as_a_judge_df['score'] = llm_as_a_judge_df['score'].fillna(fill_df_t.score)
        llm_as_a_judge_df = llm_as_a_judge_df.reset_index()
        self.eval_results_df["relevancy_llm"]=llm_as_a_judge_df.score.values
        self.eval_results["relevancy_llm"]=llm_as_a_judge_df.score.mean()
        self.response_error_list = self.response_error_list + llm_as_a_judge_df.query("score.isna()").index_num.to_list()

    def conc_llm(self):
        llm_as_a_judge_df=self.response_list_eval_df.query("eval_task=='llmcomc'")
        llm_as_a_judge_df['score']=llm_as_a_judge_df.output.apply(extract_ans,args=('score',))

        fill_df_t = self.fill_df.query("not index_num.str.contains('maskqag')").copy()
        fill_df_t['score'] = fill_df_t.output.apply(extract_ans,args=('score',))
        fill_df_t = fill_df_t.set_index('index_num')
        llm_as_a_judge_df = llm_as_a_judge_df.set_index('index_num')
        llm_as_a_judge_df['score'] = llm_as_a_judge_df['score'].fillna(fill_df_t.score)
        llm_as_a_judge_df = llm_as_a_judge_df.reset_index()
        self.eval_results_df["conc_llm"]=llm_as_a_judge_df.score.values
        self.eval_results["conc_llm"]=llm_as_a_judge_df.score.mean()
        self.response_error_list = self.response_error_list + llm_as_a_judge_df.query("score.isna()").index_num.to_list()
        
    def hal_llm(self):
        llm_as_a_judge_df=self.response_list_eval_df.query("eval_task=='llmhal'")
        llm_as_a_judge_df['score']=llm_as_a_judge_df.output.apply(extract_ans,args=('score',))
        self.eval_results_df["hal_llm"]=llm_as_a_judge_df.score.values
        self.eval_results["hal_llm"]=llm_as_a_judge_df.score.mean()

    def export_error(self,name):
        response_error_set = set(self.response_error_list)
        print(len(response_error_set))
        # 4mini
        for file in self.dirpath.glob(self.out_ext+"*test.jsonl"):
            input_list_4o_mini=get_input_openai_batch(filename_openai_input=file)
        input_list_4o_mini_error=[x for x in input_list_4o_mini if (x['custom_id'] in response_error_set)&('gpt-4o-mini' in x['body']['model'])]
        for instance in input_list_4o_mini_error:
            instance['custom_id']=instance['custom_id']+"__"+name
        
        print("error_4o_mini",len(input_list_4o_mini_error))
        input_list_other = [x for x in input_list_4o_mini if (x['custom_id'] in response_error_set)&('gpt-4-turbo' in x['body']['model'])]
        for instance in input_list_other:
            instance['custom_id']=instance['custom_id']+"__"+name
        
        print("error_4_t",len(input_list_other))
        for file in self.dirpath.glob(self.out_ext+"*test_rel.jsonl"):
            input_list_4o_rel=get_input_openai_batch(filename_openai_input=file)
        input_list_4o_rel_error=[x for x in input_list_4o_rel if x['custom_id'] in response_error_set]
        for instance in input_list_4o_rel_error:
            instance['custom_id']=instance['custom_id']+"__"+name
            instance['body']['max_tokens'] = 2048
        
        return input_list_4o_mini_error,input_list_other,input_list_4o_rel_error


eval_dict_list=[]

class eval_results():
    def __init__(self,name_list:list=[]):
        self.eval_dict_list=[]
        self.dir_dict={
            "gpt_4o":"baseline/gpt_4o",
            "gpt_4o_1shot":"baseline/gpt_4o_1shot",
            "gpt_4o_mini":"baseline/gpt_4o_mini",
            "gpt_4o_mini_1shot":"baseline/gpt_4o_mini_shot",
            "gpt_4":"baseline/gpt_4",
            "gpt_4_1shot":"baseline/gpt_4_1shot",
            "llama_3.1_8b":"baseline/llama_3.1_8b",
            "llama_3.1_8b_1shot":"baseline/llama_3.1_8b_1shot",
            "swallow_8b":"baseline/swallow_8b",
            "swallow_8b_1shot":"baseline/swallow_8b_1shot",
            "sft_swallow_instruct":"3_processed/sft_instruct",
            "sft_swallow_instruct_1shot":"3_processed/sft_instruct_1shot",
            "moe_swallow_instruct":"3_processed/moe_sft_continuous",
            "moe_swallow_merge":"3_processed/moe_swallow_merge",
            "sft_llama31_instruct":"3_processed/llama31_sft_instruct",
            "sft_llama31_instruct_1shot":"3_processed/llama31_sft_instruct_1shot",
            "sft_llama31_merge":"3_processed/llama31_sft_merge",
            "sft_llama31_merge_1shot":"3_processed/llama31_sft_merge_1shot",
            "moe_llama31_instruct":"3_processed/moe_llama31_instruct",
            "moe_llama31_merge":"3_processed/moe_llama31_merge",
            "raft_swallow":"3_processed/raft_instruct",
            "raft_swallow_p05":"3_processed/raft_swallow_p05",
            "sft_swallow_merge":"3_processed/sft_swallow_merge",
            "sft_swallow_merge_1shot":"3_processed/sft_swallow_merge_1shot",
            "sft_qwen_instruct":"3_processed/sft_qwen_instruct",
            "sft_qwen_instruct_1shot":"3_processed/sft_qwen_instruct_1shot",
            "sft_qwen_merge":"3_processed/sft_qwen_merge_instruct",
            "sft_qwen_merge_1shot":"3_processed/sft_qwen_merge_instruct_1shot/eval_cor_tokenizer",
            "qwen_2_7b":"baseline/qwen_2_7b",
            "qwen_2_7b_1shot":"baseline/qwen_2_7b_1shot",
            "raft_llama31":"3_processed/raft_llama31",
            "raft_llama31_p05":"3_processed/raft_llama31_p05",
            "raft_qwen":"3_processed/raft_qwen",
            "raft_qwen_p05":"3_processed/raft_qwen_p05",
            "moe_qwen_merge":"3_processed/moe_qwen_merge",
            "moe_qwen_merge_1shot":"3_processed/moe_qwen_merge_1shot",
            "moe_qwen_instruct":"3_processed/moe_qwen_instruct",
            "ans":"eval_data/aud_res_gpt_4o_add_ans_all",
            "llama_31_rag_invknn1":"few_shot_strategy/batch_gen_audres_many_shot_invkNN1",
            "llama_31_rag_knn2":"few_shot_strategy/batch_gen_audres_many_shot_kNN2",
            "llama_31_rag_knn5":"few_shot_strategy/batch_gen_audres_many_shot_kNN5",
            "llama_31_rag_nf2":"few_shot_strategy/batch_gen_audres_many_shot_nearfar2",
            "llama_31_rag_rand2":"few_shot_strategy/batch_gen_audres_many_shot_rand2",
            "llama_31_rag_knn3":"few_shot_strategy/batch_gen_audres_many_shot_kNN3",
            "qwen2_rag_invknn1":"few_shot_strategy_qwen2/batch_gen_audres_many_shot_invkNN1",
            "qwen2_rag_knn2":"few_shot_strategy_qwen2/batch_gen_audres_many_shot_kNN2",
            "qwen2_rag_knn3":"few_shot_strategy_qwen2/batch_gen_audres_many_shot_kNN3",
            "qwen2_rag_knn4":"few_shot_strategy_qwen2/batch_gen_audres_many_shot_kNN4",
            "qwen2_rag_knn5":"few_shot_strategy_qwen2/batch_gen_audres_many_shot_kNN5",
            "qwen2_rag_nf4":"few_shot_strategy_qwen2/batch_gen_audres_many_shot_nf4cor",
            "qwen2_rag_rand4":"few_shot_strategy_qwen2/batch_gen_audres_many_shot_rand4",
            "sft_qwen_merge_inv1shot":"3_processed/sft_qwen_merge_instruct_inv1shot",
            "sft_qwen_instruct_inv1shot":"3_processed/sft_qwen_instruct_inv1shot",
            "swallow_rag_invknn1":"few_shot_strategy_swallow/batch_gen_audres_many_shot_invkNN1",
            "swallow_rag_knn2":"few_shot_strategy_swallow/batch_gen_audres_many_shot_kNN2",
            "swallow_rag_nf2":"few_shot_strategy_swallow/batch_gen_audres_many_shot_nearfar2",
            "swallow_rag_knn3":"few_shot_strategy_swallow/batch_gen_audres_many_shot_kNN3",
            "swallow_rag_rand2":"few_shot_strategy_swallow/batch_gen_audres_many_shot_rand2",
            }
        self.out_ext_dict={
            "gpt_4o":"eval_plane_gpt_4o_",
            "gpt_4o_1shot":"eval_1shot_gpt_4o_",
            "gpt_4o_mini":"eval_plane_gpt_4o_mini_",
            "gpt_4o_mini_1shot":"eval_1shot_gpt_4o_mini_",
            "gpt_4":"eval_plane_gpt_4_",
            "gpt_4_1shot":"eval_1shot_gpt_4_",
            "llama_3.1_8b":"eval_plane_llama_3.1_8b_",
            "llama_3.1_8b_1shot":"eval_1shot_llama_3.1_8b_",
            "swallow_8b":"eval_plane_swallow_",
            "swallow_8b_1shot":"eval_plane_1shot_swallow_",
            "sft_swallow_instruct":"eval_sftinst_",
            "sft_swallow_instruct_1shot":"eval_sftinst_1shot_",
            "moe_swallow_instruct":"eval_moesft_cont_",
            "moe_swallow_merge":"eval_moe_swallow_merge_",
            "sft_llama31_instruct":"eval_sft_0shot_llama_3.1_8b_",
            "sft_llama31_instruct_1shot":"eval_sft_llama_3.1_8b_",
            "sft_llama31_merge":"eval_llama_3.1_8b_",
            "sft_llama31_merge_1shot":"eval_llama_3.1_8b_",
            "raft_swallow":"eval_raft_",
            "raft_swallow_p05":"eval_raft_swallow_p05_",
            "sft_swallow_merge":"eval_sft_swallow_merge_",
            "sft_swallow_merge_1shot":"eval_sft_swallow_merge_",
            "moe_llama31_instruct":"moe_llama31_instruct_",
            "moe_llama31_merge":"moe_llama31_merge_",
            "sft_qwen_instruct":"eval_sft_qwen_inst_",
            "sft_qwen_instruct_1shot":"eval_sft_qwen_1shot_",
            "sft_qwen_merge":"eval_sft_qwen_merge_",
            "sft_qwen_merge_1shot":"eval_sft_qwen_merge_1shot_",
            "qwen_2_7b":"eval_plane_qwen_",
            "qwen_2_7b_1shot":"eval_plane_1shot_qwen_",
            "raft_llama31":"eval_raft_llama31_",
            "raft_llama31_p05":"eval_raft_llama31_p05_",
            "raft_qwen":"eval_raft_qwen_",
            "raft_qwen_p05":"eval_raft_qwen_p05_",
            "moe_qwen_merge":"eval_qwen_moe_merge_",
            "moe_qwen_merge_1shot":"eval_qwen_moe_merge_1shot_",
            "moe_qwen_instruct":"eval_moe_instruct_",
            "ans":"aud_res_gpt_4o_add_ans_all_",
            "llama_31_rag_invknn1":"batch_gen_audres_many_shot_invkNN1_",
            "llama_31_rag_knn2":"batch_gen_audres_many_shot_kNN2_",
            "llama_31_rag_nf2":"batch_gen_audres_many_shot_nearfar2_",
            "llama_31_rag_nf4":"batch_gen_audres_many_shot_nearfar4_",
            "llama_31_rag_rand2":"batch_gen_audres_many_shot_rand2_",
            "llama_31_rag_knn3":"batch_gen_audres_many_shot_kNN3_",
            "qwen2_rag_invknn1":"batch_gen_audres_many_shot_invkNN1_",
            "qwen2_rag_knn2":"batch_gen_audres_many_shot_kNN2_",
            "qwen2_rag_knn3":"batch_gen_audres_many_shot_kNN3_",
            "qwen2_rag_knn4":"eval_batch_gen_audres_many_shot_kNN4_",
            "qwen2_rag_knn5":"batch_gen_audres_many_shot_kNN5_",
            "qwen2_rag_nf2":"batch_gen_audres_many_shot_nearfar2_",
            "qwen2_rag_nf4":"batch_gen_audres_many_shot_nf4cor_",
            "qwen2_rag_rand4":"batch_gen_audres_many_shot_rand4_",
            "sft_qwen_merge_inv1shot":"eval_sft_qwen_merge_instruct_inv1shot_",
            "sft_qwen_instruct_inv1shot":"eval_sft_qwen_instruct_inv1shot_",
            "swallow_rag_invknn1":"batch_gen_audres_many_shot_invkNN1_",
            "swallow_rag_knn2":"batch_gen_audres_many_shot_kNN2_",
            "swallow_rag_nf2":"batch_gen_audres_many_shot_nearfar2_",
            "swallow_rag_knn3":"batch_gen_audres_many_shot_kNN3_",
            "swallow_rag_rand2":"batch_gen_audres_many_shot_rand2_",
            }
        if len(name_list)==0:
            self.name_list=[
                "gpt_4o",
                "gpt_4o_1shot",
                "gpt_4",
                "gpt_4_1shot",
                "moe_swallow_merge",
                "moe_swallow_instruct",
                "moe_qwen_merge",
                "sft_swallow_instruct_1shot",
                "sft_swallow_instruct",
                "sft_llama31_instruct_1shot",
                "sft_llama31_instruct",
                "moe_llama31_instruct",
                "moe_llama31_merge",
                "llama_3.1_8b",
                "llama_3.1_8b_1shot",
                "raft_swallow",
                "raft_swallow_p05",
                "raft_llama31",
                "raft_llama31_p05",
                "raft_qwen",
                "raft_qwen_p05",
                "sft_swallow_merge",
                "sft_swallow_merge_1shot",
                "sft_llama31_merge",
                "sft_llama31_merge_1shot",
                "sft_qwen_instruct",
                "sft_qwen_instruct_1shot",
                "sft_qwen_merge",
                "sft_qwen_merge_1shot",
                "qwen_2_7b",
                "qwen_2_7b_1shot",
                "swallow_8b",
                "swallow_8b_1shot",
                "moe_qwen_instruct",
                "ans",
                "llama_31_rag_knn2",
                "llama_31_rag_nf2",
                "llama_31_rag_invknn1",
                "llama_31_rag_invknn2",
                "llama_31_rag_rand2",
                "llama_31_rag_knn3",
                "qwen2_rag_invknn1",
                "qwen2_rag_knn2",
                "qwen2_rag_knn3",
                "qwen2_rag_knn4",
                "qwen2_rag_knn5",
                "qwen2_rag_nf2",
                "qwen2_rag_nf4",
                "qwen2_rag_rand4",
                #"sft_qwen_merge_inv1shot",
                #"sft_qwen_instruct_inv1shot",
                "swallow_rag_invknn1",
                "swallow_rag_knn2",
                "swallow_rag_knn3",
                "swallow_rag_nf2",
                "swallow_rag_rand2"
                ]
        else:
            self.name_list=name_list
        self.eval_results_df=pd.DataFrame()
        self.error_list_40_mini_all = []
        self.error_list_other_all = []
        self.error_list_40_rel_all = []
    def calc_eval(self):
        for name in self.name_list:
            print(name)
            self.add_eval(name)
    
    def export_eval_results(self):
        export_df = pd.DataFrame(self.eval_dict_list).set_index('name').sort_values('llm_total_normalized',ascending=False)

        return export_df

    def add_eval(self,name:str):
        out_dir = PROJDIR / "data/3_processed/dataset_2310/downstream/" / self.dir_dict[name]
        out_ext=self.out_ext_dict[name]

        post_proc_eval_obj=post_proc_eval(
            filename_openai_rst=str(out_dir),
            filename_mask_qag_ans=str(out_dir / (out_ext+"mask_qag_ans.csv")),
            filename_conc_keyword_score=str(out_dir/(out_ext+"conc_keyword_score.csv")),
            out_ext=out_ext
            )

        eval_dict=post_proc_eval_obj.eval_and_export_results()
        eval_dict['name']=name
        self.eval_dict_list.append(eval_dict)
        eval_df=post_proc_eval_obj.eval_results_df
        eval_df['name']=name
        self.eval_results_df=pd.concat([self.eval_results_df,eval_df],axis=0)
        error_list_40_mini, error_list_other, error_list_40_rel=post_proc_eval_obj.export_error(name=name)
        self.error_list_40_mini_all = self.error_list_40_mini_all + error_list_40_mini
        self.error_list_other_all = self.error_list_other_all + error_list_other
        self.error_list_40_rel_all = self.error_list_40_rel_all + error_list_40_rel
        
    def export_error(self):
        return self.error_list_40_mini_all, self.error_list_other_all, self.error_list_40_rel_all

    def export_class_average(self,name_list:list=[]):
        score_column = ['mask_qag','comp_llm','conc_llm','relevancy_llm']
        if len(name_list)==0:
            name_list=self.name_list
        cls_info_dir = PROJDIR / "data/3_processed/dataset_2310/downstream/3_processed_trial/moe_sft"
        filename=str(cls_info_dir / "eval_moesft_15k_jp_instruct.csv")
        data_inf_df=pd.read_csv(filename,index_col=None,dtype=str)#.set_index('index_num').head(50)
        data_inf_df.cls_labels=data_inf_df.cls_labels.fillna('-1').astype(float).astype(str)
        data_inf_df[['id','cls_labels']]
        eval_df_g=pd.merge(
            self.eval_results_df[['name','c_id']+score_column],
            data_inf_df,
            left_on='c_id',
            right_on='id',
            how='left'
            )
        eval_df_g=eval_df_g.assign(cls_labels=eval_df_g.cls_labels.fillna('no_label'))
        return eval_df_g

eval_results_obj=eval_results(name_list=["gpt_4o"])
eval_results_obj.calc_eval()
eval_results_obj.export_eval_results()

# %%
eval_results_obj=eval_results()
eval_results_obj.calc_eval()
eval_results_obj.export_eval_results()
# %%
error_list_40_mini_all, error_list_other_all, error_list_40_rel_all = eval_results_obj.export_error()
# pick up error_list
print(len(error_list_40_mini_all))
print(len(error_list_other_all))
print(len(error_list_40_rel_all))
# 36
# 13
# 33

# %%
with open(PROJDIR / "data/3_processed/dataset_2310/downstream/error_retry_list3/error_list_40_mini_all2.jsonl", 'w') as f:
    for obj in error_list_40_mini_all:
        f.write(json.dumps(obj) + '\n')

with open(PROJDIR / "data/3_processed/dataset_2310/downstream/error_retry_list3/error_list_other_all2.jsonl", 'w') as f:
    for item in error_list_other_all:
        f.write(json.dumps(item) + '\n')
with open(PROJDIR / "data/3_processed/dataset_2310/downstream/error_retry_list3/error_list_40_rel_all2.jsonl", 'w') as f:
    for item in error_list_40_rel_all:
        f.write(json.dumps(item) + '\n')

# %%
eval_results_summary = eval_results_obj.eval_results_df.groupby(by='name').agg({'mask_qag':'mean','comp_llm':'mean','conc_llm':'mean','relevancy_llm':'mean'})

# %%
# save 
eval_results_summary.to_pickle(PROJDIR / "results/downstream/kam/eval_results/eval_results_summary.pkl")
eval_results_obj.eval_results_df.to_pickle(PROJDIR / "results/downstream/kam/eval_results/eval_results_df.pkl")
# %%

