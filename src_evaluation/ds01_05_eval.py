
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
PROJPATH=r"/Users/noro/Documents/Projects/XBRL_common_space_projection/"
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
    #pd.DataFrame(data)
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

    #response_list_eval_df=pd.concat([response_list_eval_df,pd.DataFrame(response_list_eval_rel)],axis=0)

    #for file in dirpath.glob("hal_batch_*output.jsonl"):
    #    response_list_eval_rel=get_results_openai_batch(filename_openai_rst=file,json=False)
    #response_list_eval_df=pd.concat([response_list_eval_df,pd.DataFrame(response_list_eval_rel)],axis=0)

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
    return response_list_eval_df
    #filename_mask_qag_ans=str(PROJDIR / "data/3_processed/dataset_2310/downstream/baseline" / "eval_plane_gpt_4o_minimask_qag_ans.csv")
    

def get_fill2():
    dirpath=Path("/Users/noro/Documents/Projects/XBRL_common_space_projection/data/3_processed/dataset_2310/downstream/error_retry_list2")
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

    #for file in dirpath.glob("hal_batch_*output.jsonl"):
    #    response_list_eval_rel=get_results_openai_batch(filename_openai_rst=file,json=False)
    #response_list_eval_df=pd.concat([response_list_eval_df,pd.DataFrame(response_list_eval_rel)],axis=0)


    response_list_eval_df=response_list_eval_df.assign(
        eval_task=response_list_eval_df.index_num.str.split("-",expand=True)[1],
        num_eval_task=response_list_eval_df.index_num.str.split("-",expand=True)[2]
        )
    return response_list_eval_df
    #filename_mask_qag_ans=str(PROJDIR / "data/3_processed/dataset_2310/downstream/baseline" / "eval_plane_gpt_4o_minimask_qag_ans.csv")
    
tmp_df = get_fill()
#tmp_df2 = get_fill2()
# %%
tmp_df#.index_num.str.split("__",expand=True)[0]#.value_counts()
tmp_df=tmp_df.query("not index_num.str.contains('maskqag')").set_index('index_num')
#tmp_df['pred']=tmp_df.output.apply(extract_ans,args=('score',))
#tmp_df.query("pred.isna()")
# %%
#tmp_df2=tmp_df2.query("not index_num.str.contains('maskqag')").set_index('index_num')
#tmp_df2['pred']=tmp_df2.output.apply(extract_ans,args=('score',))
#tmp_df2.query("pred.isna()")
#print(tmp_df.query("pred.isna()").iloc[0,:].output)
# %%
tmp_df=tmp_df.query("index_num.str.contains('maskqag')")
tmp_df['pred']=tmp_df.output.apply(extract_ans,args=('text',))
mask = tmp_df['pred']=='-'
tmp_df.loc[mask,'output']=tmp_df.loc[mask,'output']+"""" }"""
tmp_df['pred']=tmp_df.output.apply(extract_ans,args=('text',))
tmp_df.query("pred=='-'")
# %%
print(tmp_df.query("pred=='-'").iloc[0,:].output)


# %%
class post_proc_eval():
    def __init__(self,filename_openai_rst,filename_mask_qag_ans,filename_conc_keyword_score,out_ext=""):
        #filename_openai_rst=str(PROJDIR / "data/3_processed/dataset_2310/downstream/eval_data" / "gpt_4o_mini_output.jsonl")
        dirpath=Path(str(filename_openai_rst))
        self.dirpath = dirpath
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

        response_list_eval_df=pd.concat([response_list_eval_df,pd.DataFrame(response_list_eval_rel)],axis=0)

        #for file in dirpath.glob("hal_batch_*output.jsonl"):
        #    response_list_eval_rel=get_results_openai_batch(filename_openai_rst=file,json=False)
        #response_list_eval_df=pd.concat([response_list_eval_df,pd.DataFrame(response_list_eval_rel)],axis=0)

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
        
        #filename_mask_qag_ans=str(PROJDIR / "data/3_processed/dataset_2310/downstream/baseline" / "eval_plane_gpt_4o_minimask_qag_ans.csv")
        self.mask_qag_ans=pd.read_csv(filename_mask_qag_ans)        

        #filename="/Users/noro/Documents/Projects/XBRL_common_space_projection/data/3_processed/dataset_2310/downstream/baseline/gpt_4o_mini/eval_plane_gpt_4o_mini_conc_keyword_score.csv"
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
        filename="/Users/noro/Documents/Projects/XBRL_common_space_projection/data/3_processed/dataset_2310/downstream/eval_prep/nearest_score.csv"
        self.eval_results_df=pd.read_csv(filename)#.head(100)
        self.response_error_list = []
    def eval_and_export_results_hal(self):
        print("hal_llm")
        self.hal_llm()
        

    def eval_and_export_results(self):
        self.fill_df = get_fill()
        self.mask_qag()
        #print("comp_add_proc")
        #self.comp_add_proc()
        self.comp_llm()#
        #print("conc_keyword_score")
        #self.conc_keyword_score()
        self.conc_llm()#
        #print("hal_llm")
        #self.hal_llm()
        #print("answer_relevancy")
        #self.answer_relevancy()
        self.relevancy_llm()
        #self.eval_results["total"]=(
        #    self.eval_results["mask_qag"]
        #    +self.eval_results["comp_add_proc"]
        #    +self.eval_results["comp_llm"]
        #    +self.eval_results["conc_keyword_score"]
        #    +self.eval_results["conc_llm"]
        #    +self.eval_results["answer_relevancy"]
        #    +self.eval_results["relevancy_llm"]
        #    )
        self.eval_results["llm_total"]=(
            self.eval_results["mask_qag"]
            +self.eval_results["comp_llm"]
            +self.eval_results["conc_llm"]
            +self.eval_results["relevancy_llm"]
            )
        self.eval_results["mask_qag_normalized"]=(self.eval_results["mask_qag"]-np.mean(self.eval_results["mask_qag"]))/np.std(self.eval_results["mask_qag"])
        self.eval_results["comp_llm_normalized"]=(self.eval_results["comp_llm"]-np.mean(self.eval_results["comp_llm"]))/np.std(self.eval_results["comp_llm"])
        self.eval_results["conc_llm_normalized"]=(self.eval_results["conc_llm"]-np.mean(self.eval_results["conc_llm"]))/np.std(self.eval_results["conc_llm"])
        self.eval_results["relevancy_llm_normalized"]=(self.eval_results["relevancy_llm"]-np.mean(self.eval_results["relevancy_llm"]))/np.std(self.eval_results["relevancy_llm"])

        self.eval_results["llm_total_normalized"]=(
            self.eval_results["mask_qag_normalized"]
            +self.eval_results["comp_llm_normalized"]
            +self.eval_results["conc_llm_normalized"]
            +self.eval_results["relevancy_llm_normalized"]
            )/4
        return self.eval_results

    def conc_keyword_score(self):
        conc_keyword_score_df=pd.read_csv(self.filename_conc_keyword_score)
        conc_keyword_score=conc_keyword_score_df.score.mean()
        self.eval_results["conc_keyword_score"]=conc_keyword_score*5
        self.eval_results_df["conc_keyword_score"]=conc_keyword_score_df*5

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
        #self.llm_as_a_judge_df=llm_as_a_judge_df
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
        #self.llm_as_a_judge_df=llm_as_a_judge_df
        self.eval_results_df["conc_llm"]=llm_as_a_judge_df.score.values
        self.eval_results["conc_llm"]=llm_as_a_judge_df.score.mean()
        self.response_error_list = self.response_error_list + llm_as_a_judge_df.query("score.isna()").index_num.to_list()
        
    def hal_llm(self):
        llm_as_a_judge_df=self.response_list_eval_df.query("eval_task=='llmhal'")
        llm_as_a_judge_df['score']=llm_as_a_judge_df.output.apply(extract_ans,args=('score',))
        #self.llm_as_a_judge_df=llm_as_a_judge_df
        self.eval_results_df["hal_llm"]=llm_as_a_judge_df.score.values
        self.eval_results["hal_llm"]=llm_as_a_judge_df.score.mean()

    def comp_add_proc(self):
        comp_add_proc_df=self.response_list_eval_df.query("eval_task=='addcomp'")
        comp_add_proc_df=comp_add_proc_df.assign(
            val_data_num=comp_add_proc_df.num_eval_task.str.split("_",expand=True)[0],
            proc_num=comp_add_proc_df.num_eval_task.str.split("_",expand=True)[1],
            )
        comp_add_proc_df['score']=comp_add_proc_df.output.apply(extract_ans,args=('score',))
        score=comp_add_proc_df.groupby('val_data_num').score.mean().mean()
        #self.comp_add_proc_df=comp_add_proc_df
        self.eval_results_df["comp_add_proc"]=(comp_add_proc_df.groupby('val_data_num').score.mean().values)*5
        self.eval_results["comp_add_proc"]=(score)*5

    def answer_relevancy(self):
        filename=PROJDIR / "data/3_processed/dataset_2310/downstream" / "2_intermediate/llm_proc" /"audit_res_markdown_eval.csv"
        data_val_df=pd.read_csv(filename,index_col=None,dtype=str).set_index('index_num')
        
        ansrel_df=self.response_list_eval_df.query("eval_task=='ansrel'")
        ansrel_df=pd.merge(ansrel_df,data_val_df['description'],left_on='num_eval_task',right_index=True)
        ansrel_df['output_risk_list']=ansrel_df.output.apply(extract_json_dict)
        smp_score_list=[]
        for itr_index in ansrel_df.index:
            sr=ansrel_df.loc[itr_index,:]
            calc_rouge_obj=calc_rouge()
            score_list=[]
            for itr in range(len(sr.output_risk_list)):
                #print(sr.description)
                #print(list(sr.output_risk_list[itr].values())[0])
                score=calc_rouge_obj.calc_rouge_precision(sr.description,list(sr.output_risk_list[itr].values())[0])
                score_list.append(score)
            smp_score=np.max(score_list)
            #smp_score=np.mean(score_list)
            smp_score_list.append(smp_score)
        ansrel_df['score']=smp_score_list
        #self.ansrel_df=ansrel_df
        self.eval_results_df["answer_relevancy"]=ansrel_df.score.values*5
        self.eval_results["answer_relevancy"]=ansrel_df.score.mean()*5

    def export_error(self,name):
        #response_error_set = set(self.response_list_eval_df.query("status!='Success'").index_num)
        response_error_set = set(self.response_error_list)
        print(len(response_error_set))
        # 4mini
        for file in self.dirpath.glob(self.out_ext+"*test.jsonl"):
            input_list_4o_mini=get_input_openai_batch(filename_openai_input=file)
        input_list_4o_mini_error=[x for x in input_list_4o_mini if (x['custom_id'] in response_error_set)&('gpt-4o-mini' in x['body']['model'])]
        for instance in input_list_4o_mini_error:
            instance['custom_id']=instance['custom_id']+"__"+name
#            instance['body']['max_tokens'] = 2048
        
        print("error_4o_mini",len(input_list_4o_mini_error))
        input_list_other = [x for x in input_list_4o_mini if (x['custom_id'] in response_error_set)&('gpt-4-turbo' in x['body']['model'])]
        for instance in input_list_other:
            instance['custom_id']=instance['custom_id']+"__"+name
#            instance['body']['max_tokens'] = 2048
        
        print("error_4_t",len(input_list_other))
        for file in self.dirpath.glob(self.out_ext+"*test_rel.jsonl"):
            input_list_4o_rel=get_input_openai_batch(filename_openai_input=file)
        input_list_4o_rel_error=[x for x in input_list_4o_rel if x['custom_id'] in response_error_set]
        for instance in input_list_4o_rel_error:
            instance['custom_id']=instance['custom_id']+"__"+name
            instance['body']['max_tokens'] = 2048
        
        return input_list_4o_mini_error,input_list_other,input_list_4o_rel_error
        #with open(self.dirpath / self.out_ext+"error_input.jsonl", 'w') as f:
        #    for item in input_list_4o_mini_error:
        #        f.write("%s\n" % item)
        #inputから抽出


# %%



# %%
eval_dict_list=[]
# gpt 4o mini base

#out_dir = PROJDIR / "data/3_processed/dataset_2310/downstream/baseline/gpt_4o_mini"
#out_ext="eval_plane_gpt_4o_mini_"
#
#post_proc_eval_obj=post_proc_eval(
#    filename_openai_rst=str(out_dir / "batch_6731eaed2dc8819092d2f352fbc8d666_output.jsonl"),
#    filename_mask_qag_ans=str(out_dir / (out_ext+"mask_qag_ans.csv")),
#    filename_conc_keyword_score=str(out_dir/(out_ext+"conc_keyword_score.csv"))
#    )
#    
#eval_dict=post_proc_eval_obj.eval_and_export_results()
#eval_dict['name']="gpt_4o_mini_plane"
#eval_dict_list.append(eval_dict)


# %% gpt 4o mini base 1shot

#out_dir = PROJDIR / "data/3_processed/dataset_2310/downstream/baseline/gpt_4o_mini_shot"
#out_ext="eval_1shot_gpt_4o_mini_"
#
#post_proc_eval_obj=post_proc_eval(
#    filename_openai_rst=str(out_dir / "batch_6731eb2149748190ac9e7f5dad90e958_output.jsonl"),
#    filename_mask_qag_ans=str(out_dir / (out_ext+"mask_qag_ans.csv")),
#    filename_conc_keyword_score=str(out_dir/(out_ext+"conc_keyword_score.csv"))
#    )
#    
#eval_dict=post_proc_eval_obj.eval_and_export_results()
#eval_dict['name']="gpt_4o_mini_1shot"
#eval_dict_list.append(eval_dict)


# %% gpt 4o

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
            "moe_sft_cont_1shot":"3_processed/moe_sft_continuous_1shot",
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
            #"sft_qwen_merge_1shot":"3_processed/sft_qwen_merge_instruct_1shot",
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
            "llama_31_rag_nf2_cls":"few_shot_strategy/batch_gen_audres_many_shot_nearfar2_cls",
            "llama_31_rag_invknn2":"few_shot_strategy/batch_gen_audres_many_shot_invkNN2",
            "llama_31_rag_nf4":"few_shot_strategy/batch_gen_audres_many_shot_nearfar4",
            "llama_31_rag_rand2":"few_shot_strategy/batch_gen_audres_many_shot_rand2",
            "llama_31_rag_knn3":"few_shot_strategy/batch_gen_audres_many_shot_kNN3",
            "llama_31_rag_nf3":"few_shot_strategy/batch_gen_audres_many_shot_nearfar3",
            "llama_31_rag_rand3":"few_shot_strategy/batch_gen_audres_many_shot_rand3",
            "qwen2_rag_invknn1":"few_shot_strategy_qwen2/batch_gen_audres_many_shot_invkNN1",
            "qwen2_rag_knn2":"few_shot_strategy_qwen2/batch_gen_audres_many_shot_kNN2",
            "qwen2_rag_knn3":"few_shot_strategy_qwen2/batch_gen_audres_many_shot_kNN3",
            "qwen2_rag_knn4":"few_shot_strategy_qwen2/batch_gen_audres_many_shot_kNN4",
            "qwen2_rag_knn5":"few_shot_strategy_qwen2/batch_gen_audres_many_shot_kNN5",
            "qwen2_rag_nf2":"few_shot_strategy_qwen2/batch_gen_audres_many_shot_nearfar2",
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
            "moe_sft_cont_1shot":"eval_moesft_cont_",
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
            #"sft_qwen_merge_1shot_20000":"eval_sft_qwen_merge_1shot_",
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
            "llama_31_rag_knn5":"batch_gen_audres_many_shot_kNN5_",
            "llama_31_rag_nf2":"batch_gen_audres_many_shot_nearfar2_",
            "llama_31_rag_nf2_cls":"eval_batch_gen_audres_many_shot_nearfar2_cls_",
            "llama_31_rag_invknn2":"batch_gen_audres_many_shot_invkNN2_",
            "llama_31_rag_nf4":"batch_gen_audres_many_shot_nearfar4_",
            "llama_31_rag_rand2":"batch_gen_audres_many_shot_rand2_",
            "llama_31_rag_knn3":"batch_gen_audres_many_shot_kNN3_",
            "llama_31_rag_nf3":"batch_gen_audres_many_shot_nearfar3_",
            "llama_31_rag_rand3":"batch_gen_audres_many_shot_rand3_",
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
                #"moe_sft_cont_1shot",
                "moe_qwen_merge",
                #"moe_qwen_merge_1shot",
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
                #"sft_qwen_merge_1shot_20000",
                "qwen_2_7b",
                "qwen_2_7b_1shot",
                "swallow_8b",
                "swallow_8b_1shot",
                "moe_qwen_instruct",
                "ans",
                "llama_31_rag_knn2",
                #"llama_31_rag_knn5",
                "llama_31_rag_nf2",
                #"llama_31_rag_nf2_cls",
                "llama_31_rag_invknn1",
                "llama_31_rag_invknn2",
                #"llama_31_rag_nf4",
                "llama_31_rag_rand2",
                "llama_31_rag_knn3",
                #"llama_31_rag_nf3",
                #"llama_31_rag_rand3",
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
        return eval_df_g#.groupby([['cls_labels','name']])[score_column].mean().sum(axis=1).sort_values(ascending=False)

eval_results_obj=eval_results(name_list=["gpt_4o"])
eval_results_obj.calc_eval()
eval_results_obj.export_eval_results()


# %%
post_proc_eval_obj.eval_results_df.hal_llm.value_counts()



# %%
eval_results_obj=eval_results()
eval_results_obj.calc_eval()
eval_results_obj.export_eval_results()
# %%
error_list_40_mini_all, error_list_other_all, error_list_40_rel_all = eval_results_obj.export_error()
# %%
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


eval_results_obj.eval_results_df.isna().sum()





# %%
eval_tbl = eval_results_summary.copy()

eval_tbl.mask_qag = (eval_results_summary.mask_qag-eval_results_summary.mask_qag.min())/(0.545-eval_results_summary.mask_qag.min())
eval_tbl.comp_llm = (eval_results_summary.comp_llm-eval_results_summary.comp_llm.min())/(4.40-eval_results_summary.comp_llm.min())
eval_tbl.conc_llm = (eval_results_summary.conc_llm-eval_results_summary.conc_llm.min())/(5-eval_results_summary.conc_llm.min())
eval_tbl.relevancy_llm = (eval_results_summary.relevancy_llm-eval_results_summary.relevancy_llm.min())/(5-eval_results_summary.relevancy_llm.min())

eval_tbl['mean']=eval_tbl.mean(axis=1)
eval_tbl_s=eval_tbl.sort_values('mean',ascending=False)
eval_tbl_s

# %%
eval_results_obj.eval_results_df.groupby(by='c_id').agg({'mask_qag':'max','comp_llm':'max','conc_llm':'max','relevancy_llm':'max'}).mean()

# %%
#eval_results_summary.mean()
#from sklearn import preprocessing
#ss = preprocessing.StandardScaler()
##minmax = preprocessing.MinMaxScaler()
#
#eval_tbl = pd.DataFrame(
#    ss.fit_transform(eval_results_summary)*10+50,
#    index=eval_results_summary.index,
#    columns=eval_results_summary.columns
#    )
#
#eval_tbl['mean']=eval_tbl.mean(axis=1)
#eval_tbl_s=eval_tbl.sort_values('mean',ascending=False)
#eval_tbl_s

# %%
eval_results_obj.eval_results_df.head()
# %% statistical test
from scipy.stats import wilcoxon
sr_A=eval_results_obj.eval_results_df.query("name == 'moe_qwen_merge'").conc_llm
sr_B=eval_results_obj.eval_results_df.query("name == 'moe_qwen_instruct'").conc_llm
mask = sr_A.notna() & sr_B.notna()
wilcoxon(sr_A.loc[mask],sr_B.loc[mask],mode='exact',alternative='two-sided')

# %%

# %% Figure 2
model_list_q=[
    'raft_qwen',
    #'sft_qwen_merge_1shot',
    #'sft_qwen_instruct_1shot',
    'sft_qwen_instruct',
    'qwen_2_7b_1shot',
    'qwen_2_7b',
    'qwen2_rag_knn3',
    ]
model_list_q1=[
    'sft_qwen_instruct',
    'qwen_2_7b_1shot',
    'qwen_2_7b',
    ]

model_list_q2=[
    'raft_qwen',
    'sft_qwen_merge_1shot',
    'sft_qwen_instruct_1shot',
    ]

#print("Qwen")
#display(eval_tbl.query("index in @model_list_q").sort_values('mean',ascending=False))

model_list_s=[
    'raft_swallow',
    #'sft_swallow_merge_1shot',
    #'sft_swallow_instruct_1shot',
    'sft_swallow_instruct',
    'swallow_8b_1shot',
    'swallow_8b',
    ]

model_list_s1=[
    'sft_swallow_instruct',
    'swallow_8b_1shot',
    'swallow_8b',
    ]
model_list_s2=[
    'raft_swallow',
    'sft_swallow_merge_1shot',
    'sft_swallow_instruct_1shot',
    ]

model_list_l=[
    'llama_31_rag_knn2',
    'raft_llama31',
    #'sft_llama31_merge_1shot',
    #'sft_llama31_instruct_1shot',
    'sft_llama31_instruct',
    'llama_3.1_8b_1shot',
    'llama_3.1_8b',
    ]
model_list_l1=[
    
    'sft_llama31_instruct',
    'llama_3.1_8b_1shot',
    'llama_3.1_8b',
    ]
model_list_l2=[
    'raft_llama31',
    'sft_llama31_merge_1shot',
    'sft_llama31_instruct_1shot',
    ]



# %% Figure 2
import matplotlib.pyplot as plt
fig, ax = plt.subplots(3, 1, figsize=(8,8))
ax[0].plot(eval_tbl.loc[model_list_q].T,linestyle='--',marker='o')
ax[0].legend(eval_tbl.loc[model_list_q].index,loc='center left', bbox_to_anchor=(1.0, 0.5))
ax[0].title.set_text("Qwen 2 7b")
ax[1].plot(eval_tbl.loc[model_list_s].T,linestyle='--',marker='o')
ax[1].legend(eval_tbl.loc[model_list_s].index,loc='center left', bbox_to_anchor=(1.0, 0.5))
ax[1].title.set_text("Swallow 8b")
ax[2].plot(eval_tbl.loc[model_list_l].T,linestyle='--',marker='o')
ax[2].legend(eval_tbl.loc[model_list_l].index,loc='center left', bbox_to_anchor=(1.0, 0.5))
ax[2].title.set_text("Llama 3.1 8b")
fig.tight_layout()
plt.show()

fig2_text="""
- どのモデルもSFT及び1 shotでbaseから4つのメトリクスにおいて改善しており、このタスクのドメイン適応余地があることがわかる。
- SFTと1-shotを比較するとmask_qag、comp_llm、relevancy_llmにおいてSFTの方が良い結果を出している。
(2-shotではmask_qag,comp_llmが改善しSFTを上回っている)
- 3つのモデルにおいてRAFTは1shotから、さらにmask_qag,comp_llmが改善している。

+knn4,5が必要

"""

print(fig2_text)

# %% Figure 3
fig, ax = plt.subplots(3, 1, figsize=(8,8))
ax[0].plot(eval_tbl.loc[model_list_q1].T,linestyle='--',marker='o',alpha=0.3)
ax[0].plot(eval_tbl.loc[model_list_q2].T,linestyle='--',marker='o')
ax[0].legend(eval_tbl.loc[model_list_q1+model_list_q2].index,loc='center left', bbox_to_anchor=(1.0, 0.5))
ax[0].title.set_text("Qwen 2 7b")
ax[1].plot(eval_tbl.loc[model_list_s1].T,linestyle='--',marker='o',alpha=0.3)
ax[1].plot(eval_tbl.loc[model_list_s2].T,linestyle='--',marker='o')
ax[1].legend(eval_tbl.loc[model_list_s1+model_list_s2].index,loc='center left', bbox_to_anchor=(1.0, 0.5))
ax[1].title.set_text("Swallow 8b")
ax[2].plot(eval_tbl.loc[model_list_l1].T,linestyle='--',marker='o',alpha=0.3)
ax[2].plot(eval_tbl.loc[model_list_l2].T,linestyle='--',marker='o')
ax[2].legend(eval_tbl.loc[model_list_l1+model_list_l2].index,loc='center left', bbox_to_anchor=(1.0, 0.5))
ax[2].title.set_text("Llama 3.1 8b")
fig.tight_layout()
plt.show()


fig3_text="""
1 shot+ sftの比較
- raftは必ずしも(SFT,1-shot)において優れた方法とは限らないが、1 shot+SFTよりも安定して高いパフォーマンスである。
1-shot+SFTはQwenではraftよりcomp_llm, conc_llmで改善しているものの、llamaとswallowではraftに劣っており、swallowではSFTにも劣っている.
1-shot mergeはSFTで伸びるmask_qag、comp_llmで高い水準を維持しつつ、1 shotで大きく伸びるconc_llmも維持している。
"""

print(fig3_text)

# %% Figure 3


model_list_s3=[
    #'raft_swallow',
    #'swallow_8b_1shot',
    'swallow_8b',
    #'sft_swallow_merge_1shot',
    ]
model_list_s4=[
    'moe_swallow_instruct',
    'moe_swallow_merge',
    'sft_swallow_instruct',
    'sft_swallow_merge',
    ]

model_list_q3=[
    #'raft_swallow',
    #'swallow_8b_1shot',
    'qwen_2_7b',
    #'sft_swallow_merge_1shot',
    ]
model_list_q4=[
    'moe_qwen_instruct',
    'moe_qwen_merge',
    'sft_qwen_instruct',
    'sft_qwen_merge',
    ]

model_list_l3=[
    #'raft_swallow',
    #'swallow_8b_1shot',
    'llama_3.1_8b',
    #'sft_swallow_merge_1shot',
    ]
model_list_l4=[
    'moe_llama31_instruct',
    'moe_llama31_merge',
    'sft_llama31_instruct',
    'sft_llama31_merge',
    ]

#eval_tbl.loc[model_list_s2]
fig, ax = plt.subplots(3, 1, figsize=(8,8))
ax[0].plot(eval_tbl.loc[model_list_q3].T,linewidth=1,linestyle='--',marker='.',alpha=0.3)
ax[0].plot(eval_tbl.loc[model_list_q4].T,linewidth=1,linestyle='--',marker='.')
ax[0].legend(eval_tbl.loc[model_list_q3+model_list_q4].index,loc='center left', bbox_to_anchor=(1.0, 0.5))
ax[0].title.set_text("Qwen 2 7b")

ax[1].plot(eval_tbl.loc[model_list_s3].T,linewidth=1,linestyle='--',marker='.',alpha=0.3)
ax[1].plot(eval_tbl.loc[model_list_s4].T,linewidth=1,linestyle='--',marker='.')
ax[1].legend(eval_tbl.loc[model_list_s3+model_list_s4].index,loc='center left', bbox_to_anchor=(1.0, 0.5))
ax[1].title.set_text("Swallow 8b")

ax[2].plot(eval_tbl.loc[model_list_l3].T,linewidth=1,linestyle='--',marker='.',alpha=0.3)
ax[2].plot(eval_tbl.loc[model_list_l4].T,linewidth=1,linestyle='--',marker='.')
ax[2].legend(eval_tbl.loc[model_list_l3+model_list_l4].index,loc='center left', bbox_to_anchor=(1.0, 0.5))
ax[2].title.set_text("Llama 3.1 8b")
fig.tight_layout()
plt.show()


fig4_text="""
sftの継続は全体的に少し改善している.改善幅が小さいためSFTの時点で十分にdomiain全体にspecializedされていると言える。
一方でmergeの場合はあまり改善していない. mergeによって重みの改善影響が反映しにくくなってしまっている可能性がある。
"""

print(fig4_text)

# %% Figure 4 RAG





model_list_q5=[
    "qwen2_rag_knn2",
    "qwen2_rag_nf2",
    'qwen_2_7b_1shot',
    'qwen_2_7b',
    
    #'raft_swallow',
    #'swallow_8b_1shot',
    #'llama_3.1_8b',
    #'sft_swallow_merge_1shot',
    #"llama_31_rag_knn3",
    #"llama_31_rag_nf3",
    #"llama_31_rag_rand3"
    
    ]
model_list_l5=[
    #'moe_qwen_instruct',
    #'moe_qwen_merge',
    'llama_3.1_8b_1shot',
    #'sft_llama31_merge',
    "llama_31_rag_knn2",
    #"llama_31_rag_knn5",
    "llama_31_rag_nf2",
    "llama_31_rag_invknn2",
   # "llama_31_rag_nf4",
    "llama_31_rag_rand2",
    'llama_3.1_8b',
    #"llama_31_rag_rand3",
    #"llama_31_rag_knn3",
    ]

fig, ax = plt.subplots(3, 1, figsize=(8,8))

ax[0].plot(eval_tbl.loc[model_list_q5].T,linestyle='--',marker='o')
ax[0].legend(eval_tbl.loc[model_list_q5].index,loc='center left', bbox_to_anchor=(1.0, 0.5))
ax[0].title.set_text("Qwen 2")

#ax[2].plot(eval_tbl.loc[model_list_l5].T,linestyle='--',marker='o',alpha=0.3)
ax[2].plot(eval_tbl.loc[model_list_l5].T,linestyle='--',marker='o')
ax[2].legend(eval_tbl.loc[model_list_l5].index,loc='center left', bbox_to_anchor=(1.0, 0.5))
ax[2].title.set_text("Llama 3.1 8b")
fig.tight_layout()
plt.show()

display(eval_tbl.loc[model_list_l5])



# %%

model_list_q6=[
    "qwen2_rag_knn2",
    "qwen2_rag_knn3",
    'qwen_2_7b_1shot',
    'qwen_2_7b',
    ]


model_list_l6=[
    #'moe_qwen_instruct',
    #'moe_qwen_merge',
    'llama_3.1_8b_1shot',
    #'sft_llama31_merge',
    "llama_31_rag_knn2",
    "llama_31_rag_knn5",
   # "llama_31_rag_nf2",
   # "llama_31_rag_invknn2",
   # "llama_31_rag_nf4",
   # "llama_31_rag_rand2",
    'llama_3.1_8b',
   # "llama_31_rag_rand3",
    "llama_31_rag_knn3",
    ]

fig, ax = plt.subplots(3, 1, figsize=(8,8))

ax[0].plot(eval_tbl.loc[model_list_q6].T,linestyle='--',marker='o')
ax[0].legend(eval_tbl.loc[model_list_q6].index,loc='center left', bbox_to_anchor=(1.0, 0.5))
ax[0].title.set_text("Qwen 7b")


#ax[2].plot(eval_tbl.loc[model_list_l5].T,linestyle='--',marker='o',alpha=0.3)
ax[2].plot(eval_tbl.loc[model_list_l6].T,linestyle='--',marker='o')
ax[2].legend(eval_tbl.loc[model_list_l6].index,loc='center left', bbox_to_anchor=(1.0, 0.5))
ax[2].title.set_text("Llama 3.1 8b")
fig.tight_layout()
plt.show()

display(eval_tbl.loc[model_list_l6])













# %%
#df=pd.pivot_table(export_cls_df.query("name in [@model_focus,@model_base]"),index='name',columns='cls_labels',values=['mask_qag','comp_llm','conc_llm','relevancy_llm'],aggfunc='mean')
df=pd.pivot_table(export_cls_df,index='name',columns='cls_labels',values=['mask_qag','comp_llm','conc_llm','relevancy_llm'],aggfunc='mean')
#.sort_values('mask_qag',ascending=False)
df2=pd.DataFrame(
    ss.transform(df.stack()[eval_results_summary.columns]),
    index=df.stack().index,
    columns=eval_results_summary.columns
    )#.unstack().sort_values('mean',ascending=False)
#df2['average']=df2.mean(axis=1)
#df3['mean']=df3.mean(axis=1)
#.sort_values(ascending=False)
name_list_sort=[
            "raft_qwen",
            "raft_swallow",
            "raft_llama31",
            "sft_qwen_merge_1shot",
            "sft_swallow_merge_1shot",
            "sft_llama31_merge_1shot",
            #"sft_qwen_instruct_1shot",
            #"sft_swallow_instruct_1shot",
            #"sft_llama31_instruct_1shot",
            #"moe_sft_cont_1shot",
            #"gpt_4o_1shot",
            #"gpt_4_1shot",
            "qwen_2_7b_1shot",
            "swallow_8b_1shot",
            "llama_3.1_8b_1shot",
            #"moe_sft_cont",
            "sft_qwen_instruct",
            "sft_swallow_instruct",
            "sft_llama31_instruct",
            #"sft_qwen_merge_1shot_20000",
            #"gpt_4o",
            #"gpt_4",
            "qwen_2_7b",
            "swallow_8b",
            "llama_3.1_8b",
        ]

(df2.mean(axis=1)*10+50).unstack().loc[name_list_sort,cols_sort].rename(columns=topic_name_dict)
#.corr().rename(index=topic_name_dict,columns=topic_name_dict)
#.sort_values('average',ascending=False)
# %% memo
model_focus="sft_swallow_instruct"
model_base="swallow_8b_1shot"

#model_focus="sft_llama31_instruct"
#model_base="llama_3.1_8b_1shot"

#model_focus="sft_qwen_instruct"
#model_base="qwen_2_7b_1shot"

df3=(df2.mean(axis=1)*10+50).unstack()
df3_swallow=df3.T["sft_swallow_instruct"]-df3.T["swallow_8b_1shot"]
df3_swallow.name='swallow'
df3_llama=df3.T["sft_llama31_instruct"]-df3.T["llama_3.1_8b_1shot"]
df3_llama.name='llama'
df3_qwen=df3.T["sft_qwen_instruct"]-df3.T["qwen_2_7b_1shot"]
df3_qwen.name='qwen'
pd.concat([df3_swallow,df3_llama,df3_qwen],axis=1).loc[cols_sort,:].rename(index=topic_name_dict).corr(method='spearman')
#.rename(index=topic_name_dict,columns=topic_name_dict)
# %%

df=pd.pivot_table(export_cls_df,index='name',columns='cls_labels',values=['mask_qag','comp_llm','conc_llm','relevancy_llm'],aggfunc='mean')
df2=pd.DataFrame(
    ss.transform(df.stack()[eval_results_summary.columns]),
    index=df.stack().index,
    columns=eval_results_summary.columns
    )
df2=df2.mean(axis=1).unstack()
# %%
df_qwen = df2.loc[model_list_q,:]
df_qwen=df_qwen-df_qwen.loc['qwen_2_7b',:]

df_swallow = df2.loc[model_list_s,:]
df_swallow=df_swallow-df_swallow.loc['swallow_8b',:]

df_llama = df2.loc[model_list_l,:]
df_llama=df_llama-df_llama.loc['llama_3.1_8b',:]

df_all = pd.concat([df_qwen,df_swallow,df_llama],axis=0)
df_all.loc[name_list_sort,cols_sort].rename(columns=topic_name_dict)
#.stack()#.sort_values('mean',ascending=False)
#export_cls_df=eval_results_obj.export_class_average()

model_focus="moe_sft_cont"
model_base="sft_swallow_instruct"
#model_focus="sft_swallow_instruct"
#model_base="swallow_8b_1shot"

#model_focus="sft_llama31_instruct"
#model_base="llama_3.1_8b_1shot"

#model_focus="sft_qwen_instruct"
#model_base="qwen_2_7b_1shot"
#from scipy.stats import spearmanr

#r, p_value = spearmanr(df3_qwen, df3_swallow)
#(r, p_value)
# パーミュテーションテストの実装
def permutation_test(X, labels, n_permutations=1000):
    original_stat = some_statistic(X, labels)
    permuted_stats = []
    
    for _ in range(n_permutations):
        permuted_labels = np.random.permutation(labels)
        stat = some_statistic(X, permuted_labels)
        permuted_stats.append(stat)
    
    p_value = sum(s >= original_stat for s in permuted_stats) / n_permutations
    return p_value
df3.loc[['sft_swallow_instruct','sft_llama31_instruct','sft_qwen_instruct','llama_3.1_8b_1shot','swallow_8b_1shot','qwen_2_7b_1shot'],:]#.corr(method='spearman')


# %%
topic_name_dict={
    'no_label':"その他",
    '0.0':"税効果会計、グループ納税",
    '1.0':"有価証券や金融資産の評価",
    '2.0':"ソフトウェアの評価",
    '3.0':"棚卸資産の評価",#
    '4.0':"棚卸資産（販売用不動産）の評価",
    '5.0':"固定資産の減損",
    '6.0':"のれん、企業結合",
    '7.0':"不正、内部統制の不備への対応",
    '8.0':"継続企業の前提の検討",#
    '9.0':"不動産の売却",#
    '10.0':"貸倒引当金の評価",
    '11.0':"収益認識",
    '12.0':"偶発債務、偶発損失、引当金（保証損失）",#
    '13.0':"引当金（受注損失、工事損失）、見積原価の不確実性"
}
cols_sort=export_cls_df.query("name == 'gpt_4'").cls_labels.value_counts().index.to_list()#[:7]

df3_mask_qag=(df2['mask_qag'].unstack()*10+50)
df3_mask_qag_diff=df3_mask_qag.T[model_focus]-df3_mask_qag.T[model_base]
df3_mask_qag_diff.name='mask_qag'
df3_comp_llm=(df2['comp_llm'].unstack()*10+50)
df3_comp_llm_diff=df3_comp_llm.T[model_focus]-df3_comp_llm.T[model_base]
df3_comp_llm_diff.name='comp_llm'
df3_conc_llm=(df2['conc_llm'].unstack()*10+50)
df3_conc_llm_diff=df3_conc_llm.T[model_focus]-df3_conc_llm.T[model_base]
df3_conc_llm_diff.name='conc_llm'
df3_relevancy_llm=(df2['relevancy_llm'].unstack()*10+50)
df3_relevancy_llm_diff=df3_relevancy_llm.T[model_focus]-df3_relevancy_llm.T[model_base]
df3_relevancy_llm_diff.name='relevancy_llm'
pd.concat([df3_mask_qag_diff,df3_comp_llm_diff,df3_conc_llm_diff,df3_relevancy_llm_diff],axis=1).loc[cols_sort,:].rename(index=topic_name_dict).mean(axis=1)#.sort_values(ascending=False)



#df3=(df2['comp_llm'].unstack()*10+50)
#df3=(df2['conc_llm'].unstack()*10+50)
#df3=(df2['relevancy_llm'].unstack()*10+50)

#df3[cols_sort].rename(columns=topic_name_dict)#.sort_values('mean',ascending=False)

# %%
tmp=pd.DataFrame(export_cls_df.query("name == 'gpt_4'").cls_labels.value_counts().reset_index())
tmp['label']=tmp['cls_labels'].replace(topic_name_dict)
tmp
#.index.to_list()






# %%eval

def class_average(eval_df):
    score_column=['mask_qag','comp_add_proc','comp_llm','conc_keyword_score','conc_llm','answer_relevancy','relevancy_llm']
    cls_info_dir = PROJDIR / "data/3_processed/dataset_2310/downstream/3_processed_trial/moe_sft"
    filename=str(cls_info_dir / "eval_moesft_15k_jp_instruct.csv")
    data_inf_df=pd.read_csv(filename,index_col=None,dtype=str)#.set_index('index_num').head(50)
    data_inf_df.cls_labels=data_inf_df.cls_labels.fillna(-1).astype(int).astype(str)
    data_inf_df[['id','cls_labels']]
    eval_df_g=pd.merge(
        eval_df[['c_id']+score_column],
        data_inf_df,
        left_on='c_id',
        right_on='id',
        how='left'
        )
    eval_df_g=eval_df_g.assign(cls_labels=eval_df_g.cls_labels.fillna('no_label'))
    return eval_df_g.groupby('cls_labels')[score_column].mean().sum(axis=1).sort_values(ascending=False)

tmp=pd.concat([class_average(post_proc_eval_obj.eval_results_df),class_average(eval_df=post_proc_eval_obj_b.eval_results_df)],axis=1)

# %%
out_dir = PROJDIR / "data/3_processed/dataset_2310/downstream/3_processed_trial/moe_sft"
filename=str(out_dir / "eval_moesft_15k_jp_instruct.csv")
data_inf_df=pd.read_csv(filename,index_col=None)#.set_index('index_num').head(50)
data_inf_df.cls_labels=data_inf_df.cls_labels.fillna(-1).astype(int).astype(str)
data_inf_df[['id','cls_labels']]
#.value_counts()
# %%

# %%

def class_average(eval_df):
    score_column=['mask_qag','comp_add_proc','comp_llm','conc_keyword_score','conc_llm','answer_relevancy','relevancy_llm']
    out_dir = PROJDIR / "data/3_processed/dataset_2310/downstream/3_processed/moe_sft"
    filename=str(out_dir / "eval_moesft_15k_jp_instruct.csv")
    data_inf_df=pd.read_csv(filename,index_col=None,dtype=str)#.set_index('index_num').head(50)

    eval_df_g=pd.merge(
        eval_df[['c_id']+score_column],
        data_inf_df,
        left_on='c_id',
        right_on='id',
        how='left'
        )
    eval_df_g=eval_df_g.assign(cls_labels=eval_df_g.cls_labels.fillna('no_label'))
    return eval_df_g.groupby('cls_labels')[score_column].mean().sum(axis=1).sort_values(ascending=False)

tmp=pd.concat([class_average(post_proc_eval_obj.eval_results_df),class_average(eval_df=post_proc_eval_obj_b.eval_results_df)],axis=1)
tmp['diff']=tmp.iloc[:,0]-tmp.iloc[:,1]
tmp
# %%
tmp.sum()
#data_inf_df.cls_labels.value_counts()
# %%

data_inf_df

# %%
post_proc_eval_obj.eval_results


# %%
eval_df=post_proc_eval_obj.eval_results_df
eval_df[['mask_qag','comp_add_proc','comp_llm','conc_keyword_score','conc_llm','answer_relevancy','relevancy_llm']].corr()
# %%
#eval_df.conc_keyword_score.hist()
eval_df.conc_llm.value_counts()




# %% eval detail #################################################################
############################################


import matplotlib.pyplot as plt
#plt.figure(figsize=(10,10))
eval_df=post_proc_eval_obj.eval_results_df
eval_df['total']=(
    eval_df['mask_qag']+eval_df['comp_add_proc']+eval_df['comp_llm']+eval_df['conc_keyword_score']+eval_df['conc_llm']+eval_df['answer_relevancy']+eval_df['relevancy_llm'])
plt.scatter(eval_df['nearest_score'],eval_df['total'],label='mask_qag')

print(eval_df['total'].corr(eval_df['nearest_score']))

# %%
eval_df.query("nearest_score<0.94").total.mean()
# %%
eval_df.total.mean()
