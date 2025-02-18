
"""
Extract KAM from downloaded XBRL files from EDINET API2
Description of the process were in Appendix.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.append(r'/Users/noro/Documents/Projects/XBRL_common_space_projection')

import joblib
import os
from tqdm import tqdm
from time import sleep

import xml.etree.ElementTree as ET
import re
from zipfile import ZipFile

import os
from os.path import join, dirname
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModel
import torch

import warnings
warnings.filterwarnings('ignore')


PROJPATH = r"PROJECT_PATH"
PROJDIR = Path(PROJPATH)

# final version (v831 version+ additional data)
filename = PROJDIR / "dict_all_df_2407_v1012.csv"
dict_all_df = pd.read_csv(filename)

def agg_text(df):
    df = df.assign(cont_flg=df.key.str.contains('continue').astype(int))
    df = df.sort_values('cont_flg',ascending=True)
    return df.text.sum()

dict_all_df['id_tag'] = dict_all_df.id + "_" + dict_all_df.tag
dict_all_df_concat = dict_all_df.groupby(['id_tag']).apply(agg_text)
# %%
dict_all_df_concat.name = 'text'
dict_all_df_concat = dict_all_df_concat.reset_index()
dict_all_df = pd.merge(dict_all_df_concat,dict_all_df[['id_tag','tag','id','docID','context_ref','periodEnd','edinetCode']].drop_duplicates(keep='first'),left_on='id_tag',right_on='id_tag',how='left')
# %% irregular data
# irregular data were corrected manually

def df_add_S100NSFT(data_all_add):
    audit_res_text = data_all_add.query("docID=='S100NSFT'").loc['S100NSFT_FilingDateInstant_Row1Member_audit_res','text']+data_all_add.query("docID=='S100NSFT'").loc['S100NSFT_FilingDateInstant_Row3Member_audit_res','text']
    description_text = data_all_add.query("docID=='S100NSFT'").loc['S100NSFT_FilingDateInstant_Row1Member_description','text']+data_all_add.query("docID=='S100NSFT'").loc['S100NSFT_FilingDateInstant_Row2Member_description','text']
    # todo DFに格納
    df_S100NSFT=pd.DataFrame({
        'id_tag':['S100NSFT_FilingDateInstant_Row1Member_description','S100NSFT_FilingDateInstant_Row1Member_audit_res'],
        'id':['S100NSFT_FilingDateInstant_Row1Member','S100NSFT_FilingDateInstant_Row1Member'],
        'tag':['description','audit_res'],
        'text':[description_text,audit_res_text],
        'docID':['S100NSFT','S100NSFT'],
        'edinetCode':['E01656','E01656'],
        'periodEnd':['2021-12-31','2021-12-31'],
        'context_ref':['FilingDateInstant_Row1Member','FilingDateInstant_Row1Member']
        })
    return df_S100NSFT

def df_add_S100O831(data_all_add):
    df_S100O831=data_all_add.query("docID=='S100O831'").reset_index()#.loc['S100NSFT_FilingDateInstant_Row1Member_audit_res','text']+data_all_add.query("docID=='S100NSFT'").loc['S100NSFT_FilingDateInstant_Row3Member_audit_res','text']
    df_S100O831.loc[1,'id_tag']="S100O831_FilingDateInstant_Row1Member_description"
    df_S100O831.loc[1,'id']="S100O831_FilingDateInstant_Row1Member"
    df_S100O831.loc[1,'context_ref']="FilingDateInstant_Row1Member"
    return df_S100O831

data_all_add = dict_all_df.query("docID in ['S100NSFT','S100O831']").set_index('id_tag')
# %% merge irregular data

dict_all_df = pd.concat([dict_all_df.query("docID not in ['S100NSFT','S100O831']"),df_add_S100NSFT(data_all_add),df_add_S100O831(data_all_add)]).reset_index(drop=True)
# get token size

load_dotenv(verbose=True)
dotenv_path = join("/Users/noro/Documents/Projects/XBRL_common_space_projection/env/k/", '.env')
load_dotenv(dotenv_path)
token = os.environ.get("TKN_HF")

model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name,token=token)
def calc_token_size(text:str)->int:
    return len(tokenizer.convert_ids_to_tokens(tokenizer(text)['input_ids']))

dict_all_df = dict_all_df.assign(token_size_llama3_8b=dict_all_df.text.apply(calc_token_size))


# prep
data_all_pivot = dict_all_df.pivot_table(index='id',columns='tag',values='text',aggfunc='first')

data_all_token_size = dict_all_df.pivot_table(index='id',columns='tag',values='token_size_llama3_8b',aggfunc='first')
data_all_token_size.columns = 'token_size_llama3_8b_' + data_all_token_size.columns
data_all_pivot = pd.merge(data_all_pivot,data_all_token_size,left_index=True,right_index=True,how='left').reset_index()

id_edinetCode = dict_all_df.query("tag == 'description'")[['id','edinetCode','periodEnd','context_ref']]
data_all_pivot = pd.merge(data_all_pivot,id_edinetCode,left_on='id',right_on='id',how='left')

data_all_pivot = data_all_pivot.assign(edinetCode_term=data_all_pivot.edinetCode+"_"+data_all_pivot.periodEnd)
data_all_pivot.query("id.str.contains('S100NSFT') or id.str.contains('S100O831')")

# token size filter
print(len(data_all_pivot)) # 17198
data_all_pivot_f = data_all_pivot.query("token_size_llama3_8b_description<768 and token_size_llama3_8b_audit_res<=1024")

print(len(data_all_pivot_f)) # 17073
# for SFT data

import Levenshtein
import datetime
def calc_lev_dist(sr,col_name='description'):
    lev_dist=Levenshtein.distance(sr[col_name],sr[col_name+'_latest'])
    return lev_dist

data_all_pivot_f.periodEnd=pd.to_datetime(data_all_pivot_f.periodEnd)
# Latestを取得
data_all_pivot_latest = (
    data_all_pivot_f.query("periodEnd<'2024/03/31'")
    .groupby(by='edinetCode')
    .agg(
        LatestPeriodEnd=('periodEnd','max'),
        edinetCode_term=('edinetCode_term','max')
        )
    )
data_all_pivot_latest_df = data_all_pivot_f.query("edinetCode_term in @data_all_pivot_latest.edinetCode_term")
data_all_pivot_latest_df = data_all_pivot_latest_df.rename(columns={'audit_res':'audit_res_latest','description':'description_latest'})
dataset_id_set = set(data_all_pivot_latest_df.id)

# add representation (similar kam was dropped)
for term in ['2023/03/31','2022/03/31','2021/03/31','2020/03/31','2019/03/31']:
    term_str=datetime.datetime.strptime(term, '%Y/%m/%d')
    data_all_pivot_notlatest_df=data_all_pivot_f.query("id not in @dataset_id_set and periodEnd>=@term_str and periodEnd<'2024/03/31'")
    print(len(data_all_pivot_notlatest_df))
    
    data_all_pivot_notlatest_comp_df=pd.merge(data_all_pivot_notlatest_df,data_all_pivot_latest_df[['audit_res_latest','description_latest','edinetCode']],left_on='edinetCode',right_on='edinetCode',how='left')
    data_all_pivot_notlatest_comp_df=data_all_pivot_notlatest_comp_df.assign(
        dist_audit_res=data_all_pivot_notlatest_comp_df.apply(calc_lev_dist,col_name='audit_res',axis=1),
        dist_description=data_all_pivot_notlatest_comp_df.apply(calc_lev_dist,col_name='description',axis=1)
    )

    data_all_pivot_notlatest_comp_df=data_all_pivot_notlatest_comp_df.assign(
        dist_both=(data_all_pivot_notlatest_comp_df.dist_audit_res+data_all_pivot_notlatest_comp_df.dist_description)
        )
    nearesr_lev_length=data_all_pivot_notlatest_comp_df.groupby('id').agg({'dist_both':'min'})

    additional_id=set(nearesr_lev_length.query("dist_both>200").index)
    dataset_id_set=dataset_id_set|additional_id
    data_all_pivot_latest_df=data_all_pivot_f.query("id in @dataset_id_set")
    data_all_pivot_latest_df=data_all_pivot_latest_df.rename(columns={'audit_res':'audit_res_latest','description':'description_latest'})

print(data_all_pivot_latest_df.shape)

# saved updated version (all training pool)
data_all_pivot_latest_df.to_csv(PROJDIR / "data_all_pivot_1012.csv",index=False)

# load old version (training pool)
filename = PROJDIR / "data_all_pivot_0831.csv"
data_all_pivot_0831 = pd.read_csv(filename)

# save additional parts for markdown conversion
calc_0831 = set(data_all_pivot_0831.id)
data_all_pivot_latest_df.query("id not in @calc_0831").to_csv(PROJDIR / "data_all_pivot_1012_add.csv",index=False)

# save prediction data
data_all_pivot_2024 = data_all_pivot_f.query("periodEnd=='2024/03/31'")
data_all_pivot_2024.to_csv(PROJDIR /"data_all_pivot_1012_202403.csv",index=False)

