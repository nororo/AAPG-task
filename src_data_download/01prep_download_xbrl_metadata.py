#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 19:51:17 2019

@author: noro
Input: ダウンロード対象期間の開始時刻と終了時刻
Output: ダウンロードした書類(./data/Docs/XXXXX.zip)

1. 提出書類情報のメタデータを（EDINET APIのresponce）をEDINET_responce.csvに保存（５年前まで遡りループで取得している）
（5年以上前はダウンロードできても暗号化処理されており解凍できない）

memo

- References
https://srbrnote.work/archives/1100

http://itref.fc2web.com/lang/python/edinet.html

"""

# %% Requirements

import requests
import json
import pandas as pd
import numpy as np
import datetime
from time import sleep
import warnings
from tqdm import tqdm
PROJPATH=r"/Users/noro/Documents/Projects/XBRL_common_space_projection/"

import pandera as pa
from pandera.typing import DataFrame, Series

warnings.simplefilter('ignore')


class edinet_response_schima(pa.DataFrameModel):
    """
    seqNumber: 同日に提出された書類に提出時間順につく番号 YYYY/MM/DD-senCumberが提出順序情報になる
    docID: filename
    edinetCode: EDINETコード
    secCode: 証券コード
    JCN: 法人番号
    filerName: 提出者名
    fundCode: ファンドコード
    ordinanceCode: 政令コード
    formCode: 様式コード
    docTypeCode: 書類種別コード
    periodStart: 開始期間
    periodEnd: 終了期間
    submitDateTime: 書類提出日時 
    docDescription: EDINET の閲覧サイトの書類検索結果画面において、「提出書類」欄に表示される文字列
    issuerEdinetCode: 発行会社EDINETコード大量保有について発行会社のEDINETコード
    subjectEdinetCode: 公開買付けについて対象となるEDINETコード
    subsidiaryEdinetCode: 子会社のEDINETコードが出力されます。複数存在する場合(最大10個)、","(カンマ)で結合した文字列が出力
    currentReportReason: 臨報提出事由、臨時報告書の提出事由が出力されます。複数存在する場合、","(カンマ)で結合した文字列が出力
    parentDocID: 親書類管理番号
    opeDateTime: 「2-1-6 財務局職員による書類情報修正」、「2-1-7 財務局職員による書類の不開示」、磁気ディスク提出及び紙面提出を行った日時が出力
    withdrawalStatus: 取下書は"1"、取り下げられた書類は"2"、それ以外は"0"が出力
    docInfoEditStatus: 財務局職員が書類を修正した情報は"1"、修正された書類は"2"、それ以外は"0"が出力
    disclosureStatus: 財務局職員によって書類の不開示を開始した情報は"1"、不開示とされている書類は"2"、財務局職員によって書類の不開示を解除した情報は"3"、それ以外は"0"が出力
    xbrlFlag: 書類にXBRLがある場合は"1"それ以外0
    pdfFlag: 書類にPDFがある場合は"1"それ以外0
    attachDocFlag: 書類に代替書面・添付文書がある場合:1 それ以外:0
    englishDocFlag: 書類に英文ファイルがある場合1
    csvFlag: 書類にcsvがある場合1
    legalStatus: "1":縦覧中 "2":延長期間中(法定縦覧期間満了書類だが引き続き閲覧可能。) "0":閲覧期間満了(縦覧期間満了かつ延長期間なし、延長期間満了又は取下げにより閲覧できないもの。なお、不開示は含まない。)
    
    参考: 11_EDINET_API仕様書（version 2）.pdfより
    """
    seqNumber: Series[int] # 同日に提出された書類に提出時間順につく番号 YYYY/MM/DD-senCumberが提出順序情報になる
    docID: Series[str] # filename
    edinetCode: Series[str] = pa.Field(nullable=True) # EDINETコード
    secCode: Series[str] = pa.Field(nullable=True) # 証券コード
    JCN: Series[str] = pa.Field(nullable=True) # 法人番号
    filerName: Series[str] = pa.Field(nullable=True) # 提出者名
    fundCode: Series[str] = pa.Field(nullable=True) # ファンドコード
    ordinanceCode: Series[str] = pa.Field(nullable=True) # 政令コード
    formCode: Series[str] = pa.Field(nullable=True) # 様式コード
    docTypeCode: Series[str] = pa.Field(nullable=True) # 書類種別コード
    periodStart: Series[str] = pa.Field(nullable=True) # 開始期間
    periodEnd: Series[str] = pa.Field(nullable=True) # 終了期間
    submitDateTime: Series[str] = pa.Field(nullable=True) # 書類提出日時 
    docDescription: Series[str] = pa.Field(nullable=True) # EDINETの閲覧サイトの書類検索結果画面において、「提出書類」欄に表示される文字列
    issuerEdinetCode: Series[str] = pa.Field(nullable=True) # 発行会社EDINETコード 大量保有について発行会社の EDINETコード
    subjectEdinetCode: Series[str] = pa.Field(nullable=True) # 公開買付けについて対象となるEDINETコード
    subsidiaryEdinetCode: Series[str] = pa.Field(nullable=True) # 子会社の EDINET コードが出力されます。複数存在する場合(最大10個)、","(カンマ)で結合した文字列が出力
    currentReportReason: Series[str] = pa.Field(nullable=True) # 臨報提出事由、臨時報告書の提出事由が出力され ます。複数存在する場合、","(カンマ)で結合した文字列が出力
    parentDocID: Series[str] = pa.Field(nullable=True) # 親書類管理番号
    opeDateTime: Series[str] = pa.Field(nullable=True) # 「2-1-6 財務局職員による書類情報修正」、「2-1-7 財務局職員による書類の不開示」、磁気ディスク提出及び紙面提出を行った日時が出力
    withdrawalStatus: Series[str] = pa.Field(isin=['0','1','2']) # 取下書は"1"、取り下げられた書類は"2"、それ以外は"0"が出力
    docInfoEditStatus: Series[str] = pa.Field(isin=['0','1','2']) # 財務局職員が書類を修正した情報は"1"、修正された書類は"2"、それ以外は"0"が出力
    disclosureStatus: Series[str] = pa.Field(isin=['0','1','2','3']) # 財務局職員によって書類の不開示を開始した情報は"1"、不開示とされている書類は"2"、財務局職員によって書類の不開示を解除した情報は "3"、それ以外は"0"が出力
    xbrlFlag: Series[str] = pa.Field(isin=['0','1']) # 書類にXBRLがある場合は"1"、それ以外0
    pdfFlag: Series[str] = pa.Field(isin=['0','1']) # 書類にPDFがある場合は"1"、それ以外0
    attachDocFlag: Series[str] = pa.Field(isin=['0','1']) # 書類に代替書面・添付文書がある場合: 1、それ以外: 0
    englishDocFlag: Series[str] = pa.Field(isin=['0','1']) # 書類に英文ファイルがある場合: 1
    csvFlag: Series[str] = pa.Field(isin=['0','1']) # 書類にcsvがある場合1
    legalStatus: Series[str] = pa.Field(isin=['0','1','2']) # "1":縦覧中 "2":延長期間中(法定縦覧期間満了書類だが引き続き閲覧可能。) "0":閲覧期間満了(縦覧期間満了かつ延長期間なし、延長期間満了又は取下げにより閲覧できないもの。なお、不開示は含まない。)

def request_term(api_key:str, start_date_str:str='2024/07/20', end_date_str:str='2024/07/27')->list:
    start_date = datetime.datetime.strptime(start_date_str,"%Y/%m/%d")
    end_date = datetime.datetime.strptime(end_date_str,"%Y/%m/%d")

    sleep(1)
    res_results = []
    for itr in range(0,(end_date-start_date).days):

        target_date = start_date + datetime.timedelta(days=itr)
        params = {
            "date" : target_date.strftime("%Y-%m-%d"),
            "type" : 2,
            "Subscription-Key":api_key
        }
        res_results.append(get_edinet_metadata(params))
        sleep(1)
    return res_results

def get_edinet_metadata(params):
    # EDINET API version 1
    #EDINET_API_url = "https://disclosure.edinet-fsa.go.jp/api/v2/documents.json"
    # EDINET API version 2
    EDINET_API_url = "https://api.edinet-fsa.go.jp/api/v2/documents.json"
    
    retry = requests.adapters.Retry(connect=5, read=3)
    session = requests.Session()
    session.mount("http://", requests.adapters.HTTPAdapter(max_retries=retry))

    result = {"date": params['date'], "status": None, "data": None}
    res = session.get(EDINET_API_url, params=params, verify=False, timeout=(20, 30))
        
    if res.status_code == 200:
        result["status"] = "Success"
        try:
            res_parsed = json.loads(res.text)
            result["data"] = pd.read_json(json.dumps(res_parsed['results']), dtype=edinet_response_schima.to_schema().columns)
        except json.JSONDecodeError as e:
            result["status"] = f"JSON Decoding Error: {str(e)}"
            pass
        except Exception as e:
            result["status"] = f"Error: {str(e)}"
            pass
    else:
        result["status"] = f"Failure: {res.status_code}"
    
    return result

# %%
def main():
    your_api_key = "63595cf0df1a435a80c1eeeafba63e53"
    res_results = request_term(api_key=your_api_key, start_date_str='2023/09/01', end_date_str='2024/07/22')
    import joblib
    out_filename = PROJPATH+"data/0_metadata/trial/responce_meta_20230901_20240722.pkl.cmp"
    joblib.dump(res_results, out_filename, compress=True) 

    #data_list = [item['data'] for item in res_results if item['status'] == 'Success']
    #response_tbl=pd.concat(data_list).reset_index()
    #edinet_response_schima(response_tbl)

if __name__ == '__main__':
    main()

# %%
