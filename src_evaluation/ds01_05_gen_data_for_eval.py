
"""
preparation for evaluation "Accuracy"

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
PROJPATH=r"PROJECT_PATH"
PROJDIR=Path(PROJPATH)

# %%
#from openai import OpenAI
load_dotenv(verbose=True)
dotenv_path = join(Path(dirname(__file__)).parents[1] / "env" / "k", '.env')
load_dotenv(dotenv_path)

openai_api_obj=openai_api()

# %%

######################################################
#
#            Prompt
#
######################################################
from libs.compose_prompt import *
from libs.utils import *

class batch_inf_file_generator():
    def __init__(self,prompt_dict,make_prompt_func,eval_model="gpt_4o_mini"):
        self.prompt_dict=prompt_dict
        self.inf_list=[]
        self.make_prompt=make_prompt_func
        self.model_name=eval_model
        self.model_dict={
            "gpt_4o":"gpt-4o-2024-08-06", # 2023/10 I$2.5/O$10
            "gpt_4o_mini":"gpt-4o-mini-2024-07-18", # 2023/8 I$0.15/O$0.6
            "gpt_4_turbo":"gpt-4-turbo-2024-04-09", # 2023/12 I$10/O$30
            "gpt_4":"gpt-4", # 2023/12 I$30/O$60
            "gpt_3.5":"gpt-3.5-turbo-0125" # 2021/9 I$0.5/O$1.5
        }

    def insert_inf_list(self,text,itr_index_num):
        sys_prompt, usr_prompt = self.make_prompt(self.prompt_dict,text)
    
        messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": usr_prompt},
            ]
        temp={
            "custom_id": "request-"+str(itr_index_num),
            "method": "POST", "url": "/v1/chat/completions",
            "body": {
                "model": self.model_dict[self.model_name],
                "messages":messages,
                "temperature": 1,
                "max_tokens": 1024
                }}
        self.inf_list.append(temp)

    def export_list(self,out_filename):

        with open(out_filename, 'w') as file:
            for obj in self.inf_list:
                file.write(json.dumps(obj) + '\n')
    
    def print_sample(self):
        print("--- System Prompt: ---")
        print(self.inf_list[0]["body"]["messages"][0]["content"])
        print("--- User Prompt: ---")
        print(self.inf_list[0]["body"]["messages"][1]["content"])


# %%

def get_example_prompt(prompt_dict):
    filename=PROJDIR  /"audit_res_markdown_eval.csv"
    dict_df=pd.read_csv(filename)

    ans_text=dict_df.loc[1,'audit_res']
    description_text=dict_df.loc[1,'description']
    output_text="監査上の対応事項を以下のように具体的に立案します。\n\n1. **顧客検収プロセスの確認**:\n   - 売上高計上に関連する主要な取引先とのコミュニケーションを行い、顧客が検収を完了した日付を明確に確認するための証拠を収集する。具体的には、顧客からの検収書や受領書を確認し、期末に近い取引については、特に注意を払って検証を行う。\n\n2. **取引分析およびサンプリング**:\n   - 期末月に計上された売上高を対象にして、ランダムサンプルを抽出し、各取引の期間帰属の正確性を検証する。その際、取引の個別性を考慮し、検収が期末月内に実施されたか否かをチェックする。また、顧客の検収タイミングに関する過去のデータを分析し、パターンを把握することで、リスクの特定につなげる。\n\n3. **業績予想と実績の乖離の確認**:\n   - 公表した業績予想と実際の売上高との乖離を分析し、期末月における売上の計上が業績達成のプレッシャーに影響を受けていないかを検討する。特に、期末の売上高が予想及び前年同月と比較して過剰に増加していないかを確認し、異常があれば追加の調査を行う。\n\n4. **内部統制の評価**:\n   - 売上の計上に関する内部統制環境を評価する。特に、取引の承認プロセスや売上計上のタイミングに関するルールが適切に運用されているか、またその遵守状況を検証し、改善点を特定する。\n\n5. **期末月の売上高に関する特別検討**:\n   - 期末月の売上高に特に重点を置き、予想外の検収遅延や不適切な計上がないかを徹底的に調査する。具体的には、期末月の売上高が通常の月と比べて異常に高い場合、その理由を関連する取引先や取引内容を通じて徹底的に分析する。\n\n6. **外部確認の実施**:\n   - 重要な取引先に対する外部確認を行い、売上についての独立した証拠を得る。これにより、顧客による検収状況が適切に反映されているかの確認を行う。\n\n以上の対応により、期末月における売上高の期間帰属の適切性を確認し、財務諸表の信頼性を確保することを目的とします。"

    sys_prompt, usr_prompt = make_prompt_qag(prompt_dict,ans_text)
    prompt_qag = "### system\n\n" + sys_prompt + "\n\n### user\n\n" + usr_prompt
    print("----- qag prompt -----")
    print(prompt_qag)
    
    return prompt_qag

# %% step 1
filename=PROJDIR  /"audit_res_markdown_eval.csv"
dict_df=pd.read_csv(filename,index_col=None,dtype=str).set_index('index_num')
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

batch_inf_file_generator_obj=batch_inf_file_generator(
            prompt_dict=prompt_qag_1,
            make_prompt_func=make_prompt_qag)

for index_num in dict_df.index:
    ans_text=dict_df.loc[index_num,'output']
    #sys_prompt, usr_prompt = make_prompt_qag(prompt_qag,ans_text)
    batch_inf_file_generator_obj.insert_inf_list(ans_text,index_num)

out_filename=PROJDIR /"eval_extracted_process_from_ans_gpt4om.jsonl"
batch_inf_file_generator_obj.export_list(out_filename)
batch_inf_file_generator_obj.print_sample()
# %%

def get_results(filename_openai_rst):
    batch_output = pd.read_json(filename_openai_rst, orient='records', lines=True)
    ans_list=[]
    for itr_index in range(len(batch_output.response)):
        ans_t={
                'index_num':str(itr_index),
                'output':'-',
                'api_status':'-',
                'pred':'-',
                'status':'-',
                }
        output=batch_output.response[itr_index]['body']['choices'][0]['message']['content']
        #ans_t["output"]=output#["output"]
        #ans_t["api_status"]=output["status"]
        try:
            out_json_list=extract_json_dict(output)
            ans_t["output"]=out_json_list
            ans_t['status']='Success'
        except Exception as e:
            print(e)
            ans_t['status']='Failed'
        ans_list.append(ans_t)
    return ans_list
# %% step 2
# for each audit procedure
filename_openai_rst=PROJDIR /"eval_extracted_process_from_ans_gpt4om_output.jsonl"
response_list=get_results(filename_openai_rst=filename_openai_rst)

# %%
#dict_df
response_list_df=pd.DataFrame(response_list).set_index('index_num').rename(columns={'status':'prep_status','output':''})
dict_df_con=pd.concat([dict_df,response_list_df],axis=1)
dict_df_con

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

batch_inf_file_generator_obj=batch_inf_file_generator(
            prompt_dict=prompt_qag_2,
            make_prompt_func=make_prompt_qag)

index_num=0
for itr,response in enumerate(response_list):
    for itr_res,out_proc in enumerate(response['output']):
        index_str=str(itr)+'_'+str(itr_res)
        batch_inf_file_generator_obj.insert_inf_list(out_proc['監査手続'],index_str)
        index_num=index_num+1

out_filename=PROJDIR /"eval_mask_qa_from_process_gpt4om.jsonl"
batch_inf_file_generator_obj.export_list(out_filename)
batch_inf_file_generator_obj.print_sample()

# %% キーワード抽出
prompt_ext_keyword = {
    "qag_instruction": """提供された文章から、キーワードを抽出してください。""",
    #### 注意事項
    "qag_constraints": [
        "キーワードは1つの単語に限らず、一体で意味を成すひとつながり用語も選択できます。例: 連結貸借対照表を選択する場合は「連結」や「貸借対照表」ではなく「連結貸借対照表」を選択します。",
        "キーワードは例えば文章中における動作を行う主体や対象を選択できます。",
        "キーワードはできる限り特定可能性の高い固有の用語を選択してください。",
        "キーワードは文章の中でポイントになる用語を選択してください。",
        "キーワードは文章中の理由に関する記載範囲以外から選択してください。"
        ],
    "qag_output_formats": """#### 回答形式\n\nフォーマットは個別のjson形式で回答してください。\n\n{"キーワード":"(キーワード1)"}\n{"キーワード":"(キーワード2)"}""",
    #### 文章
    # ${}
    }

batch_inf_file_generator_obj=batch_inf_file_generator(
            prompt_dict=prompt_ext_keyword,
            make_prompt_func=make_prompt_qag_prep)

for index_num in dict_df.index:
    ans_text=dict_df.loc[index_num,'output']
    batch_inf_file_generator_obj.insert_inf_list(ans_text,index_num)

out_filename=PROJDIR / "data/3_processed/dataset_2310/downstream" / "eval_prep" /"eval_keywords_from_ans_gpt4.jsonl"
batch_inf_file_generator_obj.export_list(out_filename)
batch_inf_file_generator_obj.print_sample()
# %% Example


## 2 最新
"""
あなたは問題作成者です。
提供された文章の匿名性を高めるため、専門用語を1つ選択し、<MASK>に置換した文章を2通り提供してください。

#### 注意事項
 * 置換する専門用語は1つの単語に限らず、一体で意味を成すひとつながり用語も選択できます。
 * 置換する専門用語はできる限り特定可能性の高い固有の用語を選択してください。
 * 置換する専門用語は文章の中でポイントになる用語を選択してください。
 * 置換する専門用語は文章中の理由に関する記載範囲以外から選択してください。

#### 文章
売上計上日付と顧客から受領した検収書の日付を照合する統制の評価を実施する。

#### 回答形式
{"置換後の文章":"(置換後の文章1)","置換した用語":"(置換した用語1)"}
{"置換後の文章":"(置換後の文章2)","置換した用語":"(置換した用語2)"}


売上高が適切な会計期間に認識されているかを確認するため、期末月(3月)の売上取引を抽出し、顧客から受領した検収書に記載の日付と売上計上日付を照合する。
期末時点における売上取引の実在性を確認するため、当該取引にかかる債権について、顧客からの入金証憑との突合、もしくは顧客への直接確認を実施する。
期末日翌月(4月)の売上高のマイナス処理が当事業年度の売上高の修正として処理すべき取引ではないことを確認する。
"""

