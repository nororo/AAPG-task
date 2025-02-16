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
# %%
# train
# イレギュラー反映前に、こちらをベースに計算 (-> trial_llama3_1_convert_to_markdown.ipynb -> audit_res_markdown and audit_res_markdown_2)
filename=PROJDIR / "data/3_processed/dataset_2310/downstream/2_intermediate" / "data_all_pivot_0831.csv"
data_all_pivot_0831 = pd.read_csv(filename)

# イレギュラー反映後のsim除外pool選択
filename=PROJDIR / "data/3_processed/dataset_2310/downstream/2_intermediate" / "data_all_pivot_1012.csv"
data_all_pivot_0831_v2 = pd.read_csv(filename)

# %% 差異の調整
#set(data_all_pivot_0831_v2.id)-set(data_all_pivot_0831.id)

drop_set=set(data_all_pivot_0831.id)-set(data_all_pivot_0831_v2.id)
drop_set
# %% additional parts (-> trial_llama3_1_8b_gen_json_from_audit_res.ipynb -> audit_res_markdown_irr.csv)
filename=PROJDIR / "data/3_processed/dataset_2310/downstream/2_intermediate" / "data_all_pivot_1012_add.csv"
data_all_pivot_1012_add=pd.read_csv(filename)
# %% prediction (-> trial_llama3_1_8b_gen_json_from_audit_res.ipynb -> audit_res_markdown_eval.csv)
filename=PROJDIR / "data/3_processed/dataset_2310/downstream/2_intermediate" / "data_all_pivot_1012_202403.csv"
data_all_pivot_1012_202403=pd.read_csv(filename)


# %% correction
data_all_pivot_1012_cor=pd.concat([data_all_pivot_0831.query("id != 'S100NSFT_FilingDateInstant_Row1Member' and id not in @drop_set"),data_all_pivot_1012_add],axis=0)
#query("id.str.contains('S100NSFT') or id.str.contains('S100O831')")

# %%
#data_all_pivot_0831.query("id != 'S100NSFT_FilingDateInstant_Row1Member' and id not in @drop_set")
# %%
#########################################################################
#
# 1. Random split
#
#########################################################################



def split_dataset(data_train,data_validation,random_state):
    data_validation_edinetCode = set(data_validation.edinetCode.drop_duplicates().sample(500,random_state=random_state))
    
    data_validation_pivot = data_validation.query("edinetCode in @data_validation_edinetCode")
    data_train_pivot = data_train.query("edinetCode not in @data_validation_edinetCode")
    
    return data_train_pivot, data_validation_pivot


data_train,data_validation = split_dataset(data_all_pivot_1012_cor,data_all_pivot_1012_202403,0)
# %%
print(len(data_train))

print(len(data_validation))

# %%
# train pool (split train/dev in llama3_sft_qlora_1009.ipynb )
data_train.to_csv(PROJDIR / "data/3_processed/dataset_2310/downstream/3_processed/sft_data" / "data_train_1012.csv",index=False) # training(train + dev) data
# eval
data_validation.to_csv(PROJDIR / "data/3_processed/dataset_2310/downstream/3_processed/sft_data" / "data_validation_1012.csv",index=False) # training(train + dev) data




# %% markdown converted (0831 data divided rerun by error)
filename=PROJDIR / "data/3_processed/dataset_2310/downstream/2_intermediate/llm_proc" / "audit_res_markdown.csv"
audit_res_markdown=pd.read_csv(filename)
filename=PROJDIR / "data/3_processed/dataset_2310/downstream/2_intermediate/llm_proc" / "audit_res_markdown_2.csv"
audit_res_markdown_2=pd.read_csv(filename)
audit_res_markdown=pd.concat([audit_res_markdown.query("index_num<5151"),audit_res_markdown_2]).reset_index(drop=True)
#audit_res_markdown=pd.concat([audit_res_markdown.query("index_num<9999"),audit_res_markdown_2]).reset_index(drop=True)
# %%
# markdown converted (additional data(contains irr corrected and additional data))
filename=PROJDIR / "data/3_processed/dataset_2310/downstream/2_intermediate/llm_proc" / "audit_res_markdown_irr.csv"
audit_res_markdown_irr=pd.read_csv(filename)

# correction (drop irr missed data and add additional data)
audit_res_markdown_cor=pd.concat([audit_res_markdown.query("id != 'S100NSFT_FilingDateInstant_Row1Member' and id not in @drop_set"),audit_res_markdown_irr],axis=0)
print(len(audit_res_markdown_cor))
# extracted train
data_train_cor=audit_res_markdown_cor.query("id in @data_train.id")
print(len(data_train_cor))
# %%
assert set(data_train_cor.id)==set(data_train.id)
data_train_cor.to_csv(PROJDIR / "data/3_processed/dataset_2310/downstream/3_processed/sft_data" / "data_train_markdown_1012.csv",index=False) # training(train + dev) data
#600*1000/1000000*30*150

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
file_path = PROJDIR / "data/3_processed/dataset_2310/downstream/3_processed/sft_data" / "data_train_1012.csv"
data_train = pd.read_csv(file_path)
print(len(data_train))
data_train.head(1)
# %%
assertion_text = """
"""

processing_text = """
data_all_pivot_0831 - (data_all_pivot_0831 - data_all_pivot_0831_v2) + data_all_pivot_1012_add
-> data_all_pivot_1012_cor
data_all_pivot_1012_202403からedinetCodeを500個ランダムサンプリングして、eval datasetにする
data_all_pivot_1012_corから、evaldataのedinetCodeを除外
"""
header_note_txt = """
    id: docid + "_" + context_ref
    audit_res_latest:
    description_latest:

"""

ts_str = datetime.datetime.fromtimestamp(os.path.getctime(file_path)).strftime('%Y-%m-%d %H:%M:%S')
ts_now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

DataLinageJson_obj = DataLinageJson(**{
    "create_date": f'{ts_str}',
    "check_date": f'{ts_now}',
    "size": f'{os.path.getsize(file_path):,}',
    "file_path": str(file_path),
    "reader": "pandas.read_csv",
    "encoding": "utf-8",
    "input_data": {
        "data_all_pivot_0831":[str(PROJDIR / "data/3_processed/dataset_2310/downstream/2_intermediate" / "data_all_pivot_0831.csv")],
        "data_all_pivot_0831_v2":[str(PROJDIR / "data/3_processed/dataset_2310/downstream/2_intermediate" / "data_all_pivot_1012.csv")],
        "data_all_pivot_1012_add":[str(PROJDIR / "data/3_processed/dataset_2310/downstream/2_intermediate" / "data_all_pivot_1012_add.csv")],
        "data_all_pivot_1012_202403":[str(PROJDIR / "data/3_processed/dataset_2310/downstream/2_intermediate" / "data_all_pivot_1012_202403.csv")],
        },
    "input_data_providing_func": {},
    "index_name": data_train.index.name,
    "header": list(data_train.columns),
    "count": len(data_train),
    "unique_count_index": data_train.index.nunique(),
    "unique_count_header": data_train.describe(include='all').T['unique'].to_dict(),
    "example_rcd": data_train.iloc[0].to_dict(),
    "header_note": header_note_txt,
    "src": "data/ds01_03_dataset_split.py",
    "assertion": "",
    "processing": processing_text,
    "note": ""
})
DataLinageJson_obj.save()


# %%
# markdown conversion was done for v831 version (run following 2 code once)
file_path = PROJDIR / "data/3_processed/dataset_2310/downstream/3_processed/sft_data" / "data_validation_1012.csv"
data_validation = pd.read_csv(file_path)
print(len(data_validation))
data_validation.head(1)
# %%
assertion_text = """
"""

processing_text = """
data_all_pivot_0831 - (data_all_pivot_0831 - data_all_pivot_0831_v2) + data_all_pivot_1012_add
-> data_all_pivot_1012_cor
data_all_pivot_1012_202403からedinetCodeを500個ランダムサンプリングして、eval datasetにする
data_all_pivot_1012_corから、evaldataのedinetCodeを除外
"""
header_note_txt = """
    id: docid + "_" + context_ref
    audit_res_latest:
    description_latest:

"""

ts_str = datetime.datetime.fromtimestamp(os.path.getctime(file_path)).strftime('%Y-%m-%d %H:%M:%S')
ts_now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

DataLinageJson_obj = DataLinageJson(**{
    "create_date": f'{ts_str}',
    "check_date": f'{ts_now}',
    "size": f'{os.path.getsize(file_path):,}',
    "file_path": str(file_path),
    "reader": "pandas.read_csv",
    "encoding": "utf-8",
    "input_data": {
        "data_all_pivot_0831":[str(PROJDIR / "data/3_processed/dataset_2310/downstream/2_intermediate" / "data_all_pivot_0831.csv")],
        "data_all_pivot_0831_v2":[str(PROJDIR / "data/3_processed/dataset_2310/downstream/2_intermediate" / "data_all_pivot_1012.csv")],
        "data_all_pivot_1012_add":[str(PROJDIR / "data/3_processed/dataset_2310/downstream/2_intermediate" / "data_all_pivot_1012_add.csv")],
        "data_all_pivot_1012_202403":[str(PROJDIR / "data/3_processed/dataset_2310/downstream/2_intermediate" / "data_all_pivot_1012_202403.csv")],
        },
    "input_data_providing_func": {},
    "index_name": data_validation.index.name,
    "header": list(data_validation.columns),
    "count": len(data_validation),
    "unique_count_index": data_validation.index.nunique(),
    "unique_count_header": data_validation.describe(include='all').T['unique'].to_dict(),
    "example_rcd": data_validation.iloc[0].to_dict(),
    "header_note": header_note_txt,
    "src": "data/ds01_03_dataset_split.py",
    "assertion": "",
    "processing": processing_text,
    "note": ""
})
DataLinageJson_obj.save()
# %% 

file_path = PROJDIR / "data/3_processed/dataset_2310/downstream/3_processed/sft_data" / "data_train_markdown_1012.csv"
data_train_markdown = pd.read_csv(file_path)
print(len(data_train_markdown))
data_train_markdown.head(1)
# %%
assertion_text = """
assert set(data_train_cor.id)==set(data_train.id)
"""

processing_text = """
audit_res_markdown: audit_resのmarkdown化 5500くらいまでで中断
audit_res_markdown_2: audit_resのmarkdown化 後半5000
audit_res_markdown_irr: 追加データのmarkdown化
drop_set=set(data_all_pivot_0831.id)-set(data_all_pivot_0831_v2.id) 
（data_all_pivot_1012_add: 追加データ）
audit_res_markdown + audit_res_markdown_2 -> audit_res_markdown_irr部分を更新（更新範囲はdrop_set）
-> data_train_markdown_1012
"""
header_note_txt = """
    index_num:
    description: そのまま
    audit_res: markdown化前のテキスト
    output: markdown化後のテキスト
    id: docid + "_" + context_ref
"""

ts_str = datetime.datetime.fromtimestamp(os.path.getctime(file_path)).strftime('%Y-%m-%d %H:%M:%S')
ts_now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

DataLinageJson_obj = DataLinageJson(**{
    "create_date": f'{ts_str}',
    "check_date": f'{ts_now}',
    "size": f'{os.path.getsize(file_path):,}',
    "file_path": str(file_path),
    "reader": "pandas.read_csv",
    "encoding": "utf-8",
    "input_data": {
        "audit_res_markdown":[str(PROJDIR / "data/3_processed/dataset_2310/downstream/2_intermediate/llm_proc" / "audit_res_markdown.csv")],
        "audit_res_markdown_2":[str(PROJDIR / "data/3_processed/dataset_2310/downstream/2_intermediate/llm_proc" / "audit_res_markdown_2.csv")],
        "audit_res_markdown_irr":[str(PROJDIR / "data/3_processed/dataset_2310/downstream/2_intermediate/llm_proc" / "audit_res_markdown_irr.csv")],
        "data_train":[str(PROJDIR / "data/3_processed/dataset_2310/downstream/3_processed/sft_data" / "data_train_1012.csv")],
        },
    "input_data_providing_func": {},
    "index_name": data_train_markdown.index.name,
    "header": list(data_train_markdown.columns),
    "count": len(data_train_markdown),
    "unique_count_index": data_train_markdown.index.nunique(),
    "unique_count_header": data_train_markdown.describe(include='all').T['unique'].to_dict(),
    "example_rcd": data_train_markdown.iloc[0].to_dict(),
    "header_note": header_note_txt,
    "src": "data/ds01_03_dataset_split.py",
    "assertion": "",
    "processing": processing_text,
    "note": ""
})
DataLinageJson_obj.save()
# %%
