'''
parce extracted xbrl from downloaded zip file
FROM:
    data/2_intermediate/data_pool_XXXX/[NAME]/fs.xbrl

'''


# %%
#import re
import joblib
import glob
import pandas as pd
from tqdm import tqdm

from time import sleep
import sys
sys.path.append(r'/Users/noro/Documents/Projects/XBRL_common_space_projection')

from src.data import metadata_loader
from src.data import preproc_rst_loader
#from src.data.metadata_loader import get_docid_list_2020, get_responce, float_to_str, get_edinetcode
#from src.data.preproc_rst_loader import get_preproc_rst_03
# %%
import importlib
importlib.reload(metadata_loader)
importlib.reload(preproc_rst_loader)
PROJPATH=r"/Users/noro/Documents/Projects/XBRL_common_space_projection/"

# %%
'''
For the sake of define downstreem task dataset,
Concatanate following table:
    request metadata
    extracted from document
        company code
        year
        maundment label
    preprocessing status.
'''

# %% response meta data and download data(01 and 02)
'''
Get additional data to 2023 dataset from 2020 data; add non-duplicated data.
'''

filename = PROJPATH+"data/0_metadata/trial/res_results_doc_yuho_2407_v0830.pkl.cmp"
results_doc = joblib.load(filename)
# %%
len(results_doc)
# %%
results_expa_df_2023=preproc_rst_loader.get_preproc_rst_03(dataset_doc_year='yuho_2023')
contained_2023=set(results_expa_df_2023.docid)
results_expa_df_2020=preproc_rst_loader.get_preproc_rst_03(dataset_doc_year='yuho_2020')
contained_2020=set(results_expa_df_2020.docid)
docid_already_obtained=contained_2020|contained_2023
docid_add22407_list = [item['docid'] for item in results_doc if (item['status'] == 'Success' and item['docid'] not in docid_already_obtained)]
# %%
response_tbl=metadata_loader.get_responce(dataset_year='2407')
edinetcodelist=metadata_loader.get_edinetcode(dataset_year='2407')
len(response_tbl)

# %%
#docInfoEditStatus->未修正の2は削除、修正後の1の方を残す
docIDs_yuho0=response_tbl.query("docTypeCode=='120' and edinetCode in @edinetcodelist and docID in @docid_add22407_list and docInfoEditStatus!='2'")
print("Number of documents in yuho: ",len(docIDs_yuho0))
docIDs_yuho=response_tbl.query("ordinanceCode == '010' and formCode == '030000' and docID in @docid_add22407_list and docInfoEditStatus!='2'")
print("Number of documents in yuho 2: ",len(docIDs_yuho))
docIDs_yuho3=response_tbl.query("docTypeCode=='120' and docID in @docid_add22407_list and docInfoEditStatus!='2'")
print("Number of documents in yuho 3: ",len(docIDs_yuho3))

# download results (02)
download_status_latest=preproc_rst_loader.get_preproc_rst_02(dataset_doc_year='yuho_2407')
print(download_status_latest.status.value_counts())
assert download_status_latest.duplicated().sum()==0

# %%
set(docIDs_yuho0.docID)-set(docIDs_yuho.docID)
# %%
set(docIDs_yuho.docID)-set(docIDs_yuho0.docID)

# %%

assert docIDs_yuho.docID.duplicated().sum()==0
responce_yuho=docIDs_yuho.set_index('docID')
responce_yuho.columns="response_"+responce_yuho.columns

assert download_status_latest.docid.duplicated().sum()==0
download_rst_yuho=download_status_latest.set_index('docid')
download_rst_yuho.columns="download_"+download_rst_yuho.columns

preproc_summary=responce_yuho.join(download_rst_yuho,how='left')
preproc_summary.head(1)
# save at 02
# %%

out_filename=PROJPATH+"data/0_metadata/trial/download_summary_yuho_2407_v0830.csv"
preproc_summary.to_csv(out_filename,encoding='utf-8')
# %% ###############################################################
# restatement 
####################################################################
filename = PROJPATH+"data/0_metadata/trial/res_results_doc_teisei_yuho_2407.pkl.cmp"
results_doc = joblib.load(filename)

results_expa_df_2023=preproc_rst_loader.get_preproc_rst_03(dataset_doc_year='teisei_yuho_2023')
contained_2023=set(results_expa_df_2023.docid)

results_expa_df_2020=preproc_rst_loader.get_preproc_rst_03(dataset_doc_year='teisei_yuho_2020')
contained_2020=set(results_expa_df_2020.docid)

docid_already_obtained=contained_2020|contained_2023


response_tbl=metadata_loader.get_responce(dataset_year='2407')
edinetcodelist=metadata_loader.get_edinetcode(dataset_year='2407')
# 01
# exclude 2023 contained data (03)
docIDs_teisei_yuho=response_tbl.query("docTypeCode=='130' and edinetCode in @edinetcodelist and docID not in @docid_already_obtained")
print("Number of documents in teisei_yuho: ",len(docIDs_teisei_yuho))
# %%
# 02
download_status_latest_amd=preproc_rst_loader.get_preproc_rst_02(dataset_doc_year='teisei_yuho_2407')
print(download_status_latest_amd.status.value_counts())
# %% 

assert docIDs_teisei_yuho.docID.duplicated().sum()==0
responce_yuho_amd=docIDs_teisei_yuho.set_index('docID')
responce_yuho_amd.columns="response_"+responce_yuho_amd.columns

assert download_status_latest_amd.docid.duplicated().sum()==0
download_rst_amd_yuho=download_status_latest_amd.set_index('docid')
download_rst_amd_yuho.columns="download_"+download_rst_amd_yuho.columns

#assert len(responce_yuho_amd)==len(download_rst_amd_yuho)
preproc_summary_amd=responce_yuho_amd.join(download_rst_amd_yuho,how='left')
#preproc_summary[download_rst_yuho.columns]=preproc_summary[download_rst_yuho.columns].fillna()
# %%

out_filename=PROJPATH+"data/0_metadata/trial/download_summary_teisei_yuho_2407.csv"
preproc_summary_amd.to_csv(out_filename,encoding='utf-8')
# %%
