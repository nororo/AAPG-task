# %%
import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.append(r'/Users/noro/Documents/Projects/XBRL_common_space_projection')

from src.data import metadata_loader

from src.data import preproc_rst_loader
from src.data import data_utils

import joblib
import os
from tqdm import tqdm
from time import sleep

import xml.etree.ElementTree as ET
import re
from zipfile import ZipFile

import warnings
warnings.filterwarnings('ignore')


PROJPATH=r"/Users/noro/Documents/Projects/XBRL_common_space_projection/"
PROJDIR=Path(PROJPATH)

# markdown conversion was done for v831 version (run following 2 code once)
#filename=PROJDIR / "data/3_processed/dataset_2310/downstream/1_raw" / "dict_all_df_2407_v831.csv"
#old_dict_all_df=pd.read_csv(filename)

# final version (v831 version+ additional data)
filename=PROJDIR / "data/3_processed/dataset_2310/downstream/1_raw" / "dict_all_df_2407_v1012.csv"
dict_all_df=pd.read_csv(filename)


# %%
def agg_text(df):
    df=df.assign(cont_flg=df.key.str.contains('continue').astype(int))
    df=df.sort_values('cont_flg',ascending=True)
    return df.text.sum()
dict_all_df['id_tag']=dict_all_df.id+"_"+dict_all_df.tag
dict_all_df_concat=dict_all_df.groupby(['id_tag']).apply(agg_text)
# %%
dict_all_df_concat.name='text'
dict_all_df_concat=dict_all_df_concat.reset_index()
#'parse_accounting_standards'
dict_all_df=pd.merge(dict_all_df_concat,dict_all_df[['id_tag','tag','id','docID','context_ref','periodEnd','edinetCode']].drop_duplicates(keep='first'),left_on='id_tag',right_on='id_tag',how='left')
# %% irregular data

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
# %% token size
import os
from os.path import join, dirname
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModel
import torch

load_dotenv(verbose=True)
dotenv_path = join("/Users/noro/Documents/Projects/XBRL_common_space_projection/env/k/", '.env')
load_dotenv(dotenv_path)
token = os.environ.get("TKN_HF")

model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name,token=token)
def calc_token_size(text:str)->int:
    return len(tokenizer.convert_ids_to_tokens(tokenizer(text)['input_ids']))

dict_all_df = dict_all_df.assign(token_size_llama3_8b=dict_all_df.text.apply(calc_token_size))


# %% prep
data_all_pivot = dict_all_df.pivot_table(index='id',columns='tag',values='text',aggfunc='first')

data_all_token_size = dict_all_df.pivot_table(index='id',columns='tag',values='token_size_llama3_8b',aggfunc='first')
data_all_token_size.columns = 'token_size_llama3_8b_' + data_all_token_size.columns
data_all_pivot = pd.merge(data_all_pivot,data_all_token_size,left_index=True,right_index=True,how='left').reset_index()

id_edinetCode = dict_all_df.query("tag == 'description'")[['id','edinetCode','periodEnd','context_ref']]#'parse_accounting_standards'
data_all_pivot = pd.merge(data_all_pivot,id_edinetCode,left_on='id',right_on='id',how='left')#.query("edinetCode.isnull()")

data_all_pivot = data_all_pivot.assign(edinetCode_term=data_all_pivot.edinetCode+"_"+data_all_pivot.periodEnd)
data_all_pivot.query("id.str.contains('S100NSFT') or id.str.contains('S100O831')")


# %% na test 1113

audit_res_na_set=set(data_all_pivot.query("audit_res.isna()").id)
dict_all_df_na=dict_all_df.query("id in @audit_res_na_set")
# %%
set(dict_all_df_na.docID)





# %% token size filter
print(len(data_all_pivot)) # 17198
#data_all_pivot_f = data_all_pivot.query("token_size_llama3_8b_description+token_size_llama3_8b_audit_res<=2048 and token_size_llama3_8b_audit_res<=1024")
data_all_pivot_f = data_all_pivot.query("token_size_llama3_8b_description<768 and token_size_llama3_8b_audit_res<=1024")

print(len(data_all_pivot_f)) # 17073
# token_size_llama3_8b_audit_res<=1024なくても条件満たす
# %%
data_all_pivot_f.query("id.str.contains('S100NSFT') or id.str.contains('S100O831')")
#mask=data_all_pivot_0831.id.str.contains('S100NSFT')
#data_all_pivot_0831.loc[mask,:]

# %%
# %%　SFT用データ

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

data_all_pivot_latest_df.shape
# %%


# saved when run once
#.to_csv(PROJDIR / "data/3_processed/dataset_2310/downstream/0831" / "data_all_pivot_0831_S100O831.csv",index=False)

# saved updated version (all training pool)
data_all_pivot_latest_df.to_csv(PROJDIR / "data/3_processed/dataset_2310/downstream/2_intermediate" / "data_all_pivot_1012.csv",index=False)

# load old version (training pool)
filename = PROJDIR / "data/3_processed/dataset_2310/downstream/2_intermediate" / "data_all_pivot_0831.csv"
data_all_pivot_0831 = pd.read_csv(filename)

# save additional parts for markdown conversion
calc_0831 = set(data_all_pivot_0831.id)
data_all_pivot_latest_df.query("id not in @calc_0831").to_csv(PROJDIR / "data/3_processed/dataset_2310/downstream/2_intermediate" / "data_all_pivot_1012_add.csv",index=False)

# %% prediction data
data_all_pivot_2024 = data_all_pivot_f.query("periodEnd=='2024/03/31'")
data_all_pivot_2024.to_csv(PROJDIR / "data/3_processed/dataset_2310/downstream/2_intermediate" / "data_all_pivot_1012_202403.csv",index=False)
data_all_pivot_2024






# %%


# %% 2025/1/13 ###################################
#
# data lineageの作成
#
##################################################
from src.data.libs.utils import DataLinageJson
import datetime
# %% 0831 output
# markdown conversion was done for v831 version (run following 2 code once)
file_path = PROJDIR / "data/3_processed/dataset_2310/downstream/2_intermediate" / "data_all_pivot_0831.csv"
data_all_pivot_0831 = pd.read_csv(file_path)
print(len(data_all_pivot_0831))
data_all_pivot_0831.head(1)
# %%
assertion_text = """
"""

processing_text = """
1. id (id_tag) でgroupbyし、textを連結(複数のcontext_refに分かれているKAMテキストを結合するため)
2. edinetCode, periodEndを追加
3. token size 計算
4. token size filter (17198 -> 17073)
Training data作成（ボイラープレートであるため、テキストの類似性が一定以上の場合は削除（最新を優先））
5. 各edinetCodeについて2024/03/31以前の最新のKAMテキストを取得 -> data_all_pivot_latest_df
6. 2023/03/31以前に遡り、levenshtein距離が200以上の場合は追加 -> data_all_pivot_latest_df
Test data作成（v1012のみ）
(7. 2024/03/31のデータを取得 -> data_all_pivot_2024)

"""
header_note_txt = """
    tag: description or audit_res
    key: prefix:element_name
    text: KAMテキスト
    docID:
    context_ref:
    id: docid + "_" + context_ref
"""

ts_str = datetime.datetime.fromtimestamp(os.path.getctime(file_path)).strftime('%Y-%m-%d %H:%M:%S')
ts_now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

DataLinageJson_obj = DataLinageJson(**{
    "create_date": f'{ts_str}',
    "check_date": f'{ts_now}',
    "size": f'{os.path.getsize(file_path):,}',
    "file_path": str(file_path),
    "reader": "pandas.read_pickle",
    "encoding": "utf-8",
    "input_data": {
        "dict_all_df":[str(PROJDIR / "data/3_processed/dataset_2310/downstream/1_raw" / "dict_all_df_2407_v831.csv")],
        },
    "input_data_providing_func": {
        "dict_all_df":"",
        },
    "index_name": data_all_pivot_0831.index.name,
    "header": list(data_all_pivot_0831.columns),
    "count": len(data_all_pivot_0831),
    "unique_count_index": data_all_pivot_0831.index.nunique(),
    "unique_count_header": data_all_pivot_0831.describe(include='all').T['unique'].to_dict(),
    "example_rcd": data_all_pivot_0831.iloc[0].to_dict(),
    "header_note": header_note_txt,
    "src": "data/ds01_02_make_dataset_kam_generater.py",
    "assertion": "",
    "processing": processing_text,
    "note": ""
})
DataLinageJson_obj.save()


# %% 1012 output
# final version (v831 version+ additional data)
file_path = PROJDIR / "data/3_processed/dataset_2310/downstream/2_intermediate" / "data_all_pivot_1012.csv"
data_all_pivot_latest_df = pd.read_csv(file_path)
print(len(data_all_pivot_latest_df))
data_all_pivot_latest_df.head(1)
# %%

assertion_text = """
"""

ts_str = datetime.datetime.fromtimestamp(os.path.getctime(file_path)).strftime('%Y-%m-%d %H:%M:%S')
ts_now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

DataLinageJson_obj = DataLinageJson(**{
    "create_date": f'{ts_str}',
    "check_date": f'{ts_now}',
    "size": f'{os.path.getsize(file_path):,}',
    "file_path": str(file_path),
    "reader": "pandas.read_pickle",
    "encoding": "utf-8",
    "input_data": {
        "dict_all_df":[str(PROJDIR / "data/3_processed/dataset_2310/downstream/1_raw" / "dict_all_df_2407_v1012.csv")],
        },
    "input_data_providing_func": {
        "dict_all_df":"",
        },
    "index_name": data_all_pivot_latest_df.index.name,
    "header": list(data_all_pivot_latest_df.columns),
    "count": len(data_all_pivot_latest_df),
    "unique_count_index": data_all_pivot_latest_df.index.nunique(),
    "unique_count_header": data_all_pivot_latest_df.describe(include='all').T['unique'].to_dict(),
    "example_rcd": data_all_pivot_latest_df.iloc[0].to_dict(),
    "header_note": header_note_txt,
    "src": "data/ds01_02_make_dataset_kam_generater.py",
    "assertion": assertion_text,
    "processing": processing_text,
    "note": "training dataの最終バージョン. 旧データの0831からは、'S100NSFT','S100O831'のデータを修正し、0831の取得漏れデータを追加し、再サンプリングでサンプリング結果が変更している。len(set(0831) - set(1012)) = 625（markdown化しているが不要）, len(set(1012) - set(0831)) = 40（追加が必要->1012_addに保存）"
})
DataLinageJson_obj.save()

# %% 1012 output additional
# final version (v831 version+ additional data)
file_path = PROJDIR / "data/3_processed/dataset_2310/downstream/2_intermediate" / "data_all_pivot_1012_add.csv"
data_all_pivot_latest_df_add = pd.read_csv(file_path)
print(len(data_all_pivot_latest_df_add))
data_all_pivot_latest_df_add.head(1)
# %%

assertion_text = """
"""

ts_str = datetime.datetime.fromtimestamp(os.path.getctime(file_path)).strftime('%Y-%m-%d %H:%M:%S')
ts_now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

DataLinageJson_obj = DataLinageJson(**{
    "create_date": f'{ts_str}',
    "check_date": f'{ts_now}',
    "size": f'{os.path.getsize(file_path):,}',
    "file_path": str(file_path),
    "reader": "pandas.read_pickle",
    "encoding": "utf-8",
    "input_data": {
        "dict_all_df":[
            str(PROJDIR / "data/3_processed/dataset_2310/downstream/1_raw" / "dict_all_df_2407_v1012.csv"),
            str(PROJDIR / "data/3_processed/dataset_2310/downstream/2_intermediate" / "data_all_pivot_0831.csv")
        ],
        },
    "input_data_providing_func": {
        "dict_all_df":"",
        },
    "index_name": data_all_pivot_latest_df_add.index.name,
    "header": list(data_all_pivot_latest_df_add.columns),
    "count": len(data_all_pivot_latest_df_add),
    "unique_count_index": data_all_pivot_latest_df_add.index.nunique(),
    "unique_count_header": data_all_pivot_latest_df_add.describe(include='all').T['unique'].to_dict(),
    "example_rcd": data_all_pivot_latest_df_add.iloc[0].to_dict(),
    "header_note": header_note_txt,
    "src": "data/ds01_02_make_dataset_kam_generater.py",
    "assertion": assertion_text,
    "processing": processing_text,
    "note": "training dataの最終バージョンへの0831からの追加差分. 旧データの0831からは、'S100NSFT','S100O831'のデータを修正し、0831の取得漏れデータを追加し、再サンプリングでサンプリング結果が変更している。len(set(0831) - set(1012)) = 625（markdown化しているが不要）, len(set(1012) - set(0831)) = 40（追加が必要->this file）"
})
DataLinageJson_obj.save()

# %%
file_path = PROJDIR / "data/3_processed/dataset_2310/downstream/2_intermediate" / "data_all_pivot_1012_202403.csv"

data_all_pivot_latest_df_2403 = pd.read_csv(file_path)
print(len(data_all_pivot_latest_df_2403))
data_all_pivot_latest_df_2403.head(1)
# %%

assertion_text = """
"""

ts_str = datetime.datetime.fromtimestamp(os.path.getctime(file_path)).strftime('%Y-%m-%d %H:%M:%S')
ts_now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

DataLinageJson_obj = DataLinageJson(**{
    "create_date": f'{ts_str}',
    "check_date": f'{ts_now}',
    "size": f'{os.path.getsize(file_path):,}',
    "file_path": str(file_path),
    "reader": "pandas.read_pickle",
    "encoding": "utf-8",
    "input_data": {
        "dict_all_df":[str(PROJDIR / "data/3_processed/dataset_2310/downstream/1_raw" / "dict_all_df_2407_v1012.csv")],
        },
    "input_data_providing_func": {
        "dict_all_df":"",
        },
    "index_name": data_all_pivot_latest_df_2403.index.name,
    "header": list(data_all_pivot_latest_df_2403.columns),
    "count": len(data_all_pivot_latest_df_2403),
    "unique_count_index": data_all_pivot_latest_df_2403.index.nunique(),
    "unique_count_header": data_all_pivot_latest_df_2403.describe(include='all').T['unique'].to_dict(),
    "example_rcd": data_all_pivot_latest_df_2403.iloc[0].to_dict(),
    "header_note": header_note_txt,
    "src": "data/ds01_02_make_dataset_kam_generater.py",
    "assertion": assertion_text,
    "processing": processing_text,
    "note": "test dataの最終バージョン. test dataの件数がすくないため最新版のみをmarkdown化しており、差分修正は不要"
})
DataLinageJson_obj.save()

# %%
