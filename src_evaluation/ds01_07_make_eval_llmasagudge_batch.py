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

eval_list = [
#    "aud_res_gpt_4o_add_ans_all",
#    "aud_res_abs_k4_gpt_4o",
#    "aud_res_eval_rel_gpt_4o_add_ans_far_all",
#    "aud_res_gpt_4o_add_ans_2"
    "aud_res_abs_k1_gpt_4o",
    "aud_res_eval_rel_gpt_4o_add_ans_far",
    "aud_res_gpt_4o_add_ans_1"
]

for eval_name in eval_list:
    out_dir = PROJDIR / "data/3_processed/dataset_2310/downstream/eval_data"/eval_name
    out_dir.mkdir(parents=True, exist_ok=True)
    make_eval_batch(
        out_dir=str(out_dir),
        out_ext=eval_name+"_",
        input_filename=str(out_dir / (eval_name+"_output.jsonl")),
        model_type="openai",
        trial_flg=False,
        model_name_judge=model_name_judge,
        )

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

class post_proc_eval():
    def __init__(self,filename_openai_rst,filename_mask_qag_ans,filename_conc_keyword_score):
        #filename_openai_rst=str(PROJDIR / "data/3_processed/dataset_2310/downstream/eval_data" / "gpt_4o_mini_output.jsonl")
        dirpath=Path(str(filename_openai_rst))
        response_list_eval=[]
        response_list_eval_df = pd.DataFrame()
        if len(list(dirpath.glob("batch_*output.jsonl")))==0:
            print("batch_*.jsonl files are {}".format(len(list(dirpath.glob("batch_*.jsonl")))))
        else:
            for file in dirpath.glob("batch_*output.jsonl"):
                response_list_eval=response_list_eval+get_results_openai_batch(filename_openai_rst=file,json=False)

            #response_list_eval_df=pd.DataFrame(response_list_eval).query(
            #    "not index_num.str.contains('llmrel')"
            #    )
            response_list_eval_df=pd.concat([response_list_eval_df,pd.DataFrame(response_list_eval).query(
                "not index_num.str.contains('llmrel')"
                )],axis=0)
        
        if len(list(dirpath.glob("rel_batch_*output.jsonl")))!=1:
            print("rel_batch_*.jsonl files are {}".format(len(list(dirpath.glob("rel_batch_*.jsonl")))))
        else:
        
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
    
    def eval_and_export_results(self):
        print("mask_qag")
        self.mask_qag()
        #print("comp_add_proc")
        #self.comp_add_proc()
        print("comp_llm")
        self.comp_llm()#
        #print("conc_keyword_score")
        #self.conc_keyword_score()
        print("conc_llm")
        self.conc_llm()#
        #print("answer_relevancy")
        #self.answer_relevancy()
        print("relevancy_llm")
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
        calc_rouge_obj=calc_rouge()
        mask_qag_df['rouge_f']=calc_rouge_obj.eval_rouge(mask_qag_df.pred.to_list(),mask_qag_df.ans.to_list())
        mask_qag_df_g=mask_qag_df.groupby('val_data_num').agg({'rouge_f':'mean'})
        self.eval_results["mask_qag"]=mask_qag_df_g.mean().values[0]
        self.eval_results_df["mask_qag"]=mask_qag_df_g.values
    
    def comp_llm(self):
        llm_as_a_judge_df=self.response_list_eval_df.query("eval_task=='llmcomp'")
        llm_as_a_judge_df['score']=llm_as_a_judge_df.output.apply(extract_ans,args=('score',))
        #self.llm_as_a_judge_comp_llm_df=llm_as_a_judge_df
        self.eval_results_df["comp_llm"]=llm_as_a_judge_df.score.values
        self.eval_results["comp_llm"]=llm_as_a_judge_df.score.mean()

    def relevancy_llm(self):
        llm_as_a_judge_df=self.response_list_eval_df.query("eval_task=='llmrel'")
        llm_as_a_judge_df['score']=llm_as_a_judge_df.output.apply(extract_ans,args=('score',))
        #self.llm_as_a_judge_df=llm_as_a_judge_df
        self.eval_results_df["relevancy_llm"]=llm_as_a_judge_df.score.values
        self.eval_results["relevancy_llm"]=llm_as_a_judge_df.score.mean()

    def conc_llm(self):
        llm_as_a_judge_df=self.response_list_eval_df.query("eval_task=='llmcomc'")
        llm_as_a_judge_df['score']=llm_as_a_judge_df.output.apply(extract_ans,args=('score',))
        #self.llm_as_a_judge_df=llm_as_a_judge_df
        self.eval_results_df["conc_llm"]=llm_as_a_judge_df.score.values
        self.eval_results["conc_llm"]=llm_as_a_judge_df.score.mean()
        
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

# %%


# %% evaldata
eval_dict_list_evalval=[]

out_dir = PROJDIR / "data/3_processed/dataset_2310/downstream/eval_data/aud_res_abs_k4_gpt_4o"
out_ext="aud_res_abs_k4_gpt_4o_"

post_proc_eval_obj=post_proc_eval(
    filename_openai_rst=str(out_dir),# / "batch_6790d7db6ea881909c82bcdd7eff1f19_output.jsonl"),
    filename_mask_qag_ans=str(out_dir / (out_ext+"mask_qag_ans.csv")),
    filename_conc_keyword_score=str(out_dir/(out_ext+"conc_keyword_score.csv"))
    )
post_proc_eval_obj.conc_llm()
print(post_proc_eval_obj.eval_results_df.conc_llm.mean())

# %%
out_dir = PROJDIR / "data/3_processed/dataset_2310/downstream/eval_data/aud_res_abs_k1_gpt_4o"
out_ext="aud_res_abs_k1_gpt_4o_"

post_proc_eval_obj_k1=post_proc_eval(
    filename_openai_rst=str(out_dir),# / "batch_6790d7db6ea881909c82bcdd7eff1f19_output.jsonl"),
    filename_mask_qag_ans=str(out_dir / (out_ext+"mask_qag_ans.csv")),
    filename_conc_keyword_score=str(out_dir/(out_ext+"conc_keyword_score.csv"))
    )
post_proc_eval_obj_k1.conc_llm()
print(post_proc_eval_obj_k1.eval_results_df.conc_llm.mean())
# 4.85 -> 4.56 (k1) -> 4.42 (k4) -> mid?
# 具体性ok




# %%
out_dir = PROJDIR / "data/3_processed/dataset_2310/downstream/eval_data/aud_res_eval_rel_gpt_4o_add_ans_far_all"
out_ext="aud_res_eval_rel_gpt_4o_add_ans_far_all_"

post_proc_eval_obj=post_proc_eval(
    filename_openai_rst=str(out_dir),# / "batch_6790d7db6ea881909c82bcdd7eff1f19_output.jsonl"),
    filename_mask_qag_ans=str(out_dir / (out_ext+"mask_qag_ans.csv")),
    filename_conc_keyword_score=str(out_dir/(out_ext+"conc_keyword_score.csv"))
    )

post_proc_eval_obj.relevancy_llm()
print(post_proc_eval_obj.eval_results_df.relevancy_llm.mean())
# 4.39 -> 3.84 -> 2.43 (add all)
# 関連性ok
out_dir = PROJDIR / "data/3_processed/dataset_2310/downstream/eval_data/aud_res_eval_rel_gpt_4o_add_ans_far"
out_ext="aud_res_eval_rel_gpt_4o_add_ans_far_"

post_proc_eval_obj=post_proc_eval(
    filename_openai_rst=str(out_dir),# / "batch_6790d7db6ea881909c82bcdd7eff1f19_output.jsonl"),
    filename_mask_qag_ans=str(out_dir / (out_ext+"mask_qag_ans.csv")),
    filename_conc_keyword_score=str(out_dir/(out_ext+"conc_keyword_score.csv"))
    )

post_proc_eval_obj.relevancy_llm()
print(post_proc_eval_obj.eval_results_df.relevancy_llm.mean())

# %%
out_dir = PROJDIR / "data/3_processed/dataset_2310/downstream/eval_data/aud_res_gpt_4o_add_ans_1"
out_ext="aud_res_gpt_4o_add_ans_1_"

post_proc_eval_obj=post_proc_eval(
    filename_openai_rst=str(out_dir),# / "batch_6790d7db6ea881909c82bcdd7eff1f19_output.jsonl"),
    filename_mask_qag_ans=str(out_dir / (out_ext+"mask_qag_ans.csv")),
    filename_conc_keyword_score=str(out_dir/(out_ext+"conc_keyword_score.csv"))
    )

post_proc_eval_obj.comp_llm()
print(post_proc_eval_obj.eval_results_df.comp_llm.mean())
# 4.39 -> 3.84 -> 2.43 (add all)
# 関連性ok
out_dir = PROJDIR / "data/3_processed/dataset_2310/downstream/eval_data/aud_res_gpt_4o_add_ans_all"
out_ext="aud_res_gpt_4o_add_ans_all_"

post_proc_eval_obj=post_proc_eval(
    filename_openai_rst=str(out_dir),# / "batch_6790d7db6ea881909c82bcdd7eff1f19_output.jsonl"),
    filename_mask_qag_ans=str(out_dir / (out_ext+"mask_qag_ans.csv")),
    filename_conc_keyword_score=str(out_dir/(out_ext+"conc_keyword_score.csv"))
    )

post_proc_eval_obj.comp_llm()
print(post_proc_eval_obj.eval_results_df.comp_llm.mean())

# %%
pd.DataFrame(eval_dict_list_evalval).set_index('name')
#.sort_values('total',ascending=False)
# %%
"""
rag+sftもやる
comp_add_procが厳しすぎる

ans0.2はrelevancyさがっていない->few shotになってる -> farベースで評価.

absも評価するには正解データちょくせつが必要? or descriptionを薄めるとか

ragは検索が重要->検索しづらいところの評価? -> sftも下がる
topic hold out?（新しい知識） マルチホップ?
遠いところでの評価
"""






















# %% post process ppo gen



# %%
def chk_input(eval_output_obj,itr=0):
    tmp_df=pd.DataFrame(eval_output_obj.batch_inf_file_generator_obj.inf_list)
    mask=tmp_df.custom_id.str.contains("llm")
    tmp_df_t=tmp_df.loc[mask,:]    
    print(tmp_df_t.iloc[itr,:].body['messages'][0]['content'])
    print(tmp_df_t.iloc[itr,:].body['messages'][1]['content'])





# %% memo
def chk_output():
    filename_openai_rst=str(PROJDIR / "data/3_processed/dataset_2310/downstream/baseline/gpt_4o_mini" / "gpt_4o_mini_output_1110_3.jsonl")
    response_list_eval=get_results_openai_batch(filename_openai_rst=filename_openai_rst,json=False)

    response_list_eval_df=pd.DataFrame(response_list_eval)
    response_list_eval_df=response_list_eval_df.assign(
        eval_task=response_list_eval_df.index_num.str.split("-",expand=True)[1],
        num_eval_task=response_list_eval_df.index_num.str.split("-",expand=True)[2]
        )
    print(response_list_eval_df.value_counts('eval_task'))

# %%
llmcomc_df=response_list_eval_df.query("eval_task=='ansrel'")
response_list_eval_df.output.value_counts()
item=llmcomc_df.iloc[15,:].output
ans_item=list(extract_json_dict(item)[0].values())[0]
ans_item
#int(re.sub(r"\D","",str(ans_item)))
#llmcomc_df.output.value_counts()

# %%

filename=PROJDIR / "data/3_processed/dataset_2310/downstream" / "2_intermediate/llm_proc" /"audit_res_markdown_eval.csv"
data_val_df=pd.read_csv(filename,index_col=None,dtype=str).set_index('index_num')
llmcomc_df=pd.merge(llmcomc_df,data_val_df['description'],left_on='num_eval_task',right_index=True)
llmcomc_df['output_risk_list']=llmcomc_df.output.apply(extract_json_dict)
itr_index=5
sr=llmcomc_df.iloc[itr_index,:]
calc_rouge_obj=calc_rouge()
score_list=[]
for itr in range(len(sr.output_risk_list)):
    #print(sr.description)
    #print(list(sr.output_risk_list[itr].values())[0])
    score=calc_rouge_obj.calc_rouge_precision(sr.description,list(sr.output_risk_list[itr].values())[0])
    score_list.append(score)
np.max(score_list)

#calc_rouge_obj.calc_rouge_precision(sr.description,list(sr.output_risk_list[0].values())[0])


# %%

llmcomc_df=llmcomc_df.assign(
    val_data_num=llmcomc_df.num_eval_task.str.split("_",expand=True)[0],
    proc_num=llmcomc_df.num_eval_task.str.split("_",expand=True)[1],
    )
llmcomc_df['score']=llmcomc_df.output.apply(extract_ans,args=('score',))
llmcomc_df.groupby('val_data_num').score.mean().mean()



# %%
mask_qag_df=response_list_eval_df.query("eval_task=='maskqag'")

mask_qag_df=pd.merge(mask_qag_df,mask_qag_ans,left_on='index_num',right_on='index_num')
mask_qag_df=mask_qag_df.assign(
    val_data_num=mask_qag_df.num_eval_task.str.split("_",expand=True)[0],
    proc_num=mask_qag_df.num_eval_task.str.split("_",expand=True)[1],
    sample_num=mask_qag_df.num_eval_task.str.split("_",expand=True)[2]
    )

# %%
