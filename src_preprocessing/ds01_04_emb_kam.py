"""
Get vector expression of KAM text in embedding space for few-shot prompting.

"""

# %%
from transformers import AutoTokenizer, AutoModel
import torch
from torch import Tensor
import torch.nn.functional as F
from tqdm import tqdm


import csv
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from torch import Tensor
from scipy.spatial.distance import cdist

from pathlib import Path
import pandas as pd

# %%
PROJPATH = r"PROJECT_PATH"
PROJDIR = Path(PROJPATH)
filename = PROJDIR /"dict_all_df_2407_v1012.csv"

dict_all_df=pd.read_csv(filename)
dict_all_df.shape
assert dict_all_df.index.duplicated().sum()==0

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
dict_all_df=pd.merge(dict_all_df_concat,dict_all_df[['id_tag','tag','id','docID','context_ref','periodEnd','edinetCode']].drop_duplicates(keep='first'),left_on='id_tag',right_on='id_tag',how='left')

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
    df_S100O831.loc[1,'id_tag']="S100O831_FilingDateInstant_Row1Member_audit_res"
    df_S100O831.loc[1,'id']="S100O831_FilingDateInstant_Row1Member"
    df_S100O831.loc[1,'context_ref']="FilingDateInstant_Row1Member"
    return df_S100O831

data_all_add = dict_all_df.query("docID in ['S100NSFT','S100O831']").set_index('id_tag')
dict_all_df=pd.concat([dict_all_df.query("docID not in ['S100NSFT','S100O831']"),df_add_S100NSFT(data_all_add),df_add_S100O831(data_all_add)]).reset_index(drop=True)
dict_all_df.periodEnd=pd.to_datetime(dict_all_df.periodEnd)
dict_all_df=dict_all_df.query("periodEnd<='2024/3/31'")

dict_all_df.periodEnd=dict_all_df.periodEnd.astype(str)

data_all_pivot = dict_all_df.pivot_table(index='id',columns='tag',values='text',aggfunc='first')
id_edinetCode = dict_all_df.query("tag == 'description'")[['id','edinetCode','periodEnd','context_ref']]#'parse_accounting_standards'
data_all_pivot = pd.merge(data_all_pivot,id_edinetCode,left_on='id',right_on='id',how='left')#.query("edinetCode.isnull()")
data_all_pivot = data_all_pivot.assign(edinetCode_term=data_all_pivot.edinetCode+"_"+data_all_pivot.periodEnd)
data_all_pivot.head(1)

data_all_pivot.to_csv(PROJDIR / "data_all17k_pivot_2407_v1012.csv",index=False)

# embedding
model_name = "intfloat/multilingual-e5-large"
TOKENIZER = AutoTokenizer.from_pretrained(model_name)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL = AutoModel.from_pretrained(model_name).to(DEVICE)
def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

def cosine_similarity(v1, v2):
    return 1 - cdist([v1], [v2], 'cosine')[0][0]

def emb(text):
    inputs = TOKENIZER(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(DEVICE)
    with torch.no_grad():
        outputs = MODEL(**inputs)
    embeddings = average_pool(outputs.last_hidden_state, inputs['attention_mask'])
    embeddings = F.normalize(embeddings, p=2, dim=1)
    return embeddings[0].cpu().numpy()
# check nan
print(data_all_pivot.description.isna().sum())
# %%
vectors=data_all_pivot.description.apply(emb)
vectors_df=pd.DataFrame(vectors.to_list())
data_all_pivot = dict_all_df.pivot_table(index='id',columns='tag',values='text',aggfunc='first')
data_all_pivot.description.isna().sum()
out_filename=PROJDIR / "dict_all_df_2407_v1012_vect_df.pkl"
vectors_df.to_pickle(out_filename)

# %%
