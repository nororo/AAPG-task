
'''
2. EDINETコードリストで業種を絞って、ＥＤＩＮＥＴコードを取得　（「提出者業種」と「上場区分」でフィルター）
EDINETコードリストは以下より
https://disclosure.edinet-fsa.go.jp/E01EW/BLMainController.jsp?uji.verb=W1E62071InitDisplay&uji.bean=ee.bean.W1E62071.EEW1E62071Bean&TID=W1E62071&PID=currentPage&SESSIONKEY=1562597359623&kbn=2&ken=0&res=0&idx=0&start=null&end=null&spf1=1&spf2=1&spf5=1&psr=1&pid=0&row=100&str=&flg=&lgKbn=2&pkbn=0&skbn=1&dskb=&askb=&dflg=0&iflg=0&preId=1

3. EDINET_responce.csvをＥＤＩＮＥＴコード（edinetCode）とdocTypeCode(=120:有価証券報告書 or 140:四半期報告書)でフィルターし、ダウンロードするDocIDを取得
4. EDINET APIのURLをDocIDで構成し、zipファイルをダウンロード

'''

# %% Requirements

import requests
import json
import pandas as pd
import numpy as np
import datetime
from time import sleep
import warnings
from tqdm import tqdm
import joblib
from pathlib import Path

# %%
PROJPATH=r"/Users/noro/Documents/Projects/XBRL_common_space_projection/"
PROJDIR = Path(PROJPATH)

# %%
import urllib3
from urllib3.exceptions import InsecureRequestWarning
urllib3.disable_warnings(InsecureRequestWarning)

def request_doc(api_key,docid,out_filename_path):
    # EDINET API version 2
    EDINET_API_url = "https://api.edinet-fsa.go.jp/api/v2/documents/" + docid
    retry = requests.adapters.Retry(connect=5, read=3)
    session = requests.Session()
    session.mount("http://", requests.adapters.HTTPAdapter(max_retries=retry))

    params = {
        "type": 1, # 1:xbrl # 2: PDF 5:csv,
        "Subscription-Key": api_key
    }
    result = {"docid": docid, "status": None, "data_path": None}
    try:
        res = session.get(EDINET_API_url, params=params, verify=False, timeout=(20, 90))
        if res.status_code == 200:
            result["status"] = "Success"
            with open(out_filename_path, 'wb') as f:
                for chunk in res.iter_content(chunk_size=1024):
                    f.write(chunk)
            result["data_path"] = out_filename_path
        else:
            result["status"] = f"Failure: {res.status_code}"
    except Exception as e:
        result["status"] = f"Error: {str(e)}"
        pass
    return result

#def GetDocEdinet(docid,out_filename_path):
#    params = {
#        "type" : 1 # 1:xbrl # 2: PDF 5:csv
#    }
#    result = {"docid": docid, "status": None, "data_path": None}
#    url = 'https://disclosure.edinet-fsa.go.jp/api/v1/documents/' + docid
#    try:
#        res = requests.get(url, params=params, verify=False, timeout=(20, 90))
#        if res.status_code == 200:
#            result["status"] = "Success"
#            with open(out_filename_path, 'wb') as f:
#                for chunk in res.iter_content(chunk_size=1024):
#                    f.write(chunk)
#            result["data_path"]=out_filename_path
#        else:
#            result["status"] = f"Failure: {res.status_code}"
#    except:
#        result["status"] = "Zip Write Error"
#        pass
#    return result
# %%
def main():
    filename=PROJPATH+"data/0_metadata/trial/responce_meta_20230901_20240722.pkl.cmp"
    results_1 = joblib.load(filename)
    results_all=results_1#+results_2
    data_list = [item['data'] for item in results_all if item['status'] == 'Success']
    response_tbl=pd.concat(data_list).reset_index()

    #edinetcode=pd.read_csv(PROJPATH+'data/0_metadata/trial/EdinetcodeDlInfo2407.csv', \
    #                    header=1,index_col=False, engine="python", encoding="cp932")
    #mask=edinetcode.loc[:,'上場区分']=='上場'
    #edinetcode_filterd=edinetcode.loc[mask,:].copy()
    #edinetcodelist=edinetcode_filterd['ＥＤＩＮＥＴコード'].to_list()

    docIDs_teisei_yuho=response_tbl.query("docTypeCode=='130' and ordinanceCode == '010' and formCode == '030001' and docInfoEditStatus !='2'")
    docIDs_yuho=response_tbl.query("docTypeCode=='120' and ordinanceCode == '010' and formCode == '030000' and docInfoEditStatus !='2'")
    print("Number of documents in teisei_yuho: ",len(docIDs_teisei_yuho))
    print("Number of documents in yuho: ",len(docIDs_yuho))
    

    your_api_key = "63595cf0df1a435a80c1eeeafba63e53"
    # teisei yuho
    res_results_teisei_yuho = []
    (PROJDIR / "data/1_raw/data_pool_2407_130").mkdir(parents=True, exist_ok=True)
    for docid in tqdm(docIDs_teisei_yuho['docID']):
        out_filename = PROJDIR / "data/1_raw/data_pool_2407_130" / (docid + ".zip")
        res_results_teisei_yuho.append(request_doc(api_key=your_api_key, docid=docid, out_filename_path=out_filename))
        sleep(1)

    out_filename = PROJPATH+"data/0_metadata/trial/res_results_doc_teisei_yuho_2407_v0830.pkl.cmp"
    joblib.dump(res_results_teisei_yuho, out_filename, compress=True) 

    # yuho
    res_results_yuho=[]
    (PROJDIR / "data/1_raw/data_pool_2407_120").mkdir(parents=True, exist_ok=True)
    for docid in tqdm(docIDs_yuho['docID']):
        out_filename = PROJDIR / "data/1_raw/data_pool_2407_120" / (docid + ".zip")
        res_results_yuho.append(request_doc(api_key=your_api_key, docid=docid, out_filename_path=out_filename))
        sleep(1)

    out_filename=PROJPATH+"data/0_metadata/trial/res_results_doc_yuho_2407_v0830.pkl.cmp"
    joblib.dump(res_results_yuho, out_filename, compress=True) 

if __name__ == "__main__":
    main()
# %%
#res_results_teisei_shihanki=[]
#for docid in tqdm(docIDs_teisei_shihanki['docID']):
#    out_filename=PROJPATH+'data/1_raw/data_pool_2310_150/' + docid + '.zip'
#    res_results_teisei_shihanki.append(GetDocEdinet(docid,out_filename))
#    sleep(1)
#
#out_filename=PROJPATH+"data/0_metadata/trial/res_results_doc_teisei_shihanki.pkl.cmp"
#joblib.dump(res_results_teisei_shihanki, out_filename, compress=True) 
#
#
#
#res_results_shihanki=[]
#for docid in tqdm(docIDs_shihanki['docID']):
#    out_filename=PROJPATH+'data/1_raw/data_pool_2310_140/' + docid + '.zip'
#    res_results_shihanki.append(GetDocEdinet(docid,out_filename))
#    sleep(1)
#
#out_filename=PROJPATH+"data/0_metadata/trial/res_results_doc_shihanki.pkl.cmp"
#joblib.dump(res_results_shihanki, out_filename, compress=True) 

# %%

#data_list_retry = [item['docid'] for item in res_results_yuho if item['status'] != 'Success']

# %%

