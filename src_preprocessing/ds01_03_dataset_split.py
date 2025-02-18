# %%
import pandas as pd
import numpy as np
from pathlib import Path
import sys

import joblib
import os
from tqdm import tqdm
from time import sleep

import xml.etree.ElementTree as ET
import re
from zipfile import ZipFile

import warnings
warnings.filterwarnings('ignore')


PROJPATH=r"PROJECT_PATH"
PROJDIR=Path(PROJPATH)
# %%
# train
# イレギュラー反映前に、こちらをベースに計算 (-> trial_llama3_1_convert_to_markdown.ipynb -> audit_res_markdown and audit_res_markdown_2)
filename=PROJDIR / "data_all_pivot_0831.csv"
data_all_pivot_0831 = pd.read_csv(filename)

# イレギュラー反映後のsim除外pool選択
filename=PROJDIR /"data_all_pivot_1012.csv"
data_all_pivot_0831_v2 = pd.read_csv(filename)

# %% check preprocessed dataset difference
#set(data_all_pivot_0831_v2.id)-set(data_all_pivot_0831.id)

drop_set=set(data_all_pivot_0831.id)-set(data_all_pivot_0831_v2.id)
# %% additional parts (-> trial_llama3_1_8b_gen_json_from_audit_res.ipynb -> audit_res_markdown_irr.csv)
filename=PROJDIR / "data/3_processed/dataset_2310/downstream/2_intermediate" / "data_all_pivot_1012_add.csv"
data_all_pivot_1012_add=pd.read_csv(filename)
# %% prediction (-> trial_llama3_1_8b_gen_json_from_audit_res.ipynb -> audit_res_markdown_eval.csv)
filename=PROJDIR / "data/3_processed/dataset_2310/downstream/2_intermediate" / "data_all_pivot_1012_202403.csv"
data_all_pivot_1012_202403=pd.read_csv(filename)


# %% correction
data_all_pivot_1012_cor=pd.concat([data_all_pivot_0831.query("id != 'S100NSFT_FilingDateInstant_Row1Member' and id not in @drop_set"),data_all_pivot_1012_add],axis=0)

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
data_train.to_csv(PROJDIR /  "data_train_1012.csv",index=False) # training(train + dev) data
# eval
data_validation.to_csv(PROJDIR / "data_validation_1012.csv",index=False) # training(train + dev) data

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

