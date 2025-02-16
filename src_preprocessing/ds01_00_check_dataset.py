processing_text = """
Input 1: response_tbl
```python
response_tbl_1=metadata_loader.get_responce(dataset_year='2020')
response_tbl_2=metadata_loader.get_responce(dataset_year='2023')
response_tbl_3=metadata_loader.get_responce(dataset_year='2407')
# after concat, query
query(ordinanceCode == '010' and formCode == '030000' and docInfoEditStatus !='2' and withdrawalStatus=='0')
```
Input2: download_summary
```python
download_summary = preproc_rst_loader.download_summary(dataset_doc_year='yuho_2023_v0826')
download_summary_2020 = preproc_rst_loader.download_summary(dataset_doc_year='yuho_2020_v0830')
download_summary_2407 = preproc_rst_loader.download_summary(dataset_doc_year='yuho_2407')

download_summary = pd.concat([download_summary,download_summary_2020,download_summary_2407],axis=0)

download_summary = download_summary.query("response_withdrawalStatus!=2").reset_index().drop_duplicates(subset=['docID'],keep='last').set_index('docID')
```
proc 1: 
```python
response left joined download
```
"""

assertion_text = """
``` python
docid_diff=res_docid-down_docid
assert len(docid_diff)==0
```
"""

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

download_summary=preproc_rst_loader.download_summary(dataset_doc_year='yuho_2023_v0826')
download_summary_2020=preproc_rst_loader.download_summary(dataset_doc_year='yuho_2020_v0830')
download_summary_2407=preproc_rst_loader.download_summary(dataset_doc_year='yuho_2407')
download_summary=pd.concat([download_summary,download_summary_2020,download_summary_2407],axis=0)
download_summary=download_summary.query("response_withdrawalStatus!=2").reset_index().drop_duplicates(subset=['docID'],keep='last').set_index('docID')
download_summary.head(1)


# %%
response_tbl_1=metadata_loader.get_responce(dataset_year='2020')
response_tbl_2=metadata_loader.get_responce(dataset_year='2023')
response_tbl_3=metadata_loader.get_responce(dataset_year='2407')
# %%
import pandera as pa
from pandera.typing import DataFrame, Series
def get_columns(schima:pa.DataFrameModel)->list:
    return list(schima.to_schema().columns.keys())

class edinet_response_schima(pa.DataFrameModel):
    """
    docID: filename
    edinetCode: EDINETコード
    secCode: 証券コード
    filerName: 提出者名
    fundCode: ファンドコード
    ordinanceCode: 政令コード
    formCode: 様式コード
    docTypeCode: 書類種別コード
    periodStart: 開始期間
    periodEnd: 終了期間
    withdrawalStatus: 取下書は"1"、取り下げられた書類は"2"、それ以外は"0"が出力
    docInfoEditStatus: 財務局職員が書類を修正した情報は"1"、修正された書類は"2"、それ以外は"0"が出力
    参考: 11_EDINET_API仕様書（version 2）.pdfより
    """
    docID: Series[str] # filename
    edinetCode: Series[str] = pa.Field(nullable=True) # EDINETコード
    secCode: Series[str] = pa.Field(nullable=True) # 証券コード
    filerName: Series[str] = pa.Field(nullable=True) # 提出者名
    fundCode: Series[str] = pa.Field(nullable=True) # ファンドコード
    ordinanceCode: Series[str] = pa.Field(nullable=True) # 政令コード
    formCode: Series[str] = pa.Field(nullable=True) # 様式コード
    docTypeCode: Series[str] = pa.Field(nullable=True) # 書類種別コード
    periodStart: Series[str] = pa.Field(nullable=True) # 開始期間
    periodEnd: Series[str] = pa.Field(nullable=True) # 終了期間
    #subsidiaryEdinetCode: Series[str] = pa.Field(nullable=True) # 子会社の EDINET コードが出力されます。複数存在する場合(最大10個)、","(カンマ)で結合した文字列が出力
    withdrawalStatus: Series[str] = pa.Field(isin=['0','1','2']) # 取下書は"1"、取り下げられた書類は"2"、それ以外は"0"が出力
    docInfoEditStatus: Series[str] = pa.Field(isin=['0','1','2']) # 財務局職員が書類を修正した情報は"1"、修正された書類は"2"、それ以外は"0"が出力
    
def float_to_str(series):
    return series.astype('str').str.replace('\.0','').replace('nan', np.nan)

print(" === prep response 1 ===")

response_tbl_1['edinetCode']=response_tbl_1.edinetCode.fillna('-')
response_tbl_1['filerName']=response_tbl_1.filerName.fillna('-')

response_tbl_1['withdrawalStatus']=response_tbl_1.withdrawalStatus.astype(str)
response_tbl_1['docInfoEditStatus']=response_tbl_1.docInfoEditStatus.astype(str)
response_tbl_1['secCode']=response_tbl_1.secCode.str.replace('.0','').replace('nan', np.nan).fillna('-')
response_tbl_1['fundCode']=response_tbl_1.fundCode.fillna('-')
response_tbl_1['ordinanceCode']=response_tbl_1.ordinanceCode.str.replace('.0','').str.zfill(3).replace('-00', np.nan).fillna('-')
response_tbl_1['formCode']=response_tbl_1.formCode.str.replace('.0','').str.zfill(6).replace('-00000', np.nan).fillna('-')
response_tbl_1['docTypeCode']=response_tbl_1.docTypeCode.str.replace('.0','').replace('nan', np.nan).fillna('-')
response_tbl_1['periodStart']=response_tbl_1.periodStart.fillna('-')
response_tbl_1['periodEnd']=response_tbl_1.periodEnd.fillna('-')
response_tbl_1v=edinet_response_schima(response_tbl_1[get_columns(edinet_response_schima)])
# %%
print(" === prep response 2 ===")
response_tbl_2['edinetCode']=response_tbl_2.edinetCode.fillna('-')
response_tbl_2['filerName']=response_tbl_2.filerName.fillna('-')

response_tbl_2['withdrawalStatus']=response_tbl_2.withdrawalStatus.astype(str)
response_tbl_2['docInfoEditStatus']=response_tbl_2.docInfoEditStatus.astype(str)
response_tbl_2['secCode']=response_tbl_2.secCode.str.replace('.0','').replace('nan', np.nan).fillna('-')
response_tbl_2['fundCode']=response_tbl_2.fundCode.fillna('-')
response_tbl_2['ordinanceCode']=response_tbl_2.ordinanceCode.str.replace('.0','').str.zfill(3).replace('-00', np.nan).fillna('-')
response_tbl_2['formCode']=response_tbl_2.formCode.str.replace('.0','').str.zfill(6).replace('-00000', np.nan).fillna('-')
response_tbl_2['docTypeCode']=response_tbl_2.docTypeCode.str.replace('.0','').replace('nan', np.nan).fillna('-')
response_tbl_2['periodStart']=response_tbl_2.periodStart.fillna('-')
response_tbl_2['periodEnd']=response_tbl_2.periodEnd.fillna('-')

response_tbl_2v=edinet_response_schima(response_tbl_2[get_columns(edinet_response_schima)])
# %%
print(" === prep response 3 ===")
response_tbl_3['edinetCode']=response_tbl_3.edinetCode.fillna('-')
response_tbl_3['filerName']=response_tbl_3.filerName.fillna('-')
response_tbl_3['secCode']=response_tbl_3.secCode.fillna('-')
response_tbl_3['fundCode']=response_tbl_3.fundCode.fillna('-')
response_tbl_3['periodStart']=response_tbl_3.periodStart.fillna('-')
response_tbl_3['periodEnd']=response_tbl_3.periodEnd.fillna('-')

response_tbl_3v=edinet_response_schima(response_tbl_3[get_columns(edinet_response_schima)])

# %% 
print(" === concat response_tbl ===")
response_tbl_concat=pd.concat([
    response_tbl_1v.query("ordinanceCode == '010' and formCode == '030000' and docInfoEditStatus !='2' and withdrawalStatus=='0'"),
    response_tbl_2v.query("ordinanceCode == '010' and formCode == '030000' and docInfoEditStatus !='2' and withdrawalStatus=='0'"),
    response_tbl_3v.query("ordinanceCode == '010' and formCode == '030000' and docInfoEditStatus !='2' and withdrawalStatus=='0'")
],axis=0)
# %%
print(" === drop duplicates on response_tbl ===")
response_tbl_concat=response_tbl_concat.drop_duplicates(subset=['docID'],keep='last')
# %%
print(" === response set - download set ===")
down_docid=set(download_summary.index)
res_docid=set(response_tbl_concat.docID)
docid_diff=res_docid-down_docid
print("res - download size:",len(docid_diff))
assert len(docid_diff)==0
# %%
print(" === download set - res set ===")
docid_diff=down_docid-res_docid
print("down - res size:",len(docid_diff))
print("不要なものもダウンロードしている?")

# %%
print(" === Download漏れがないことを確認 ===")
down_docid=set(download_summary.index)
res_docid=set(response_tbl_concat.docID)
docid_diff=res_docid-down_docid
assert len(response_tbl_concat.query("docID in @docid_diff"))==0
response_tbl_concat=response_tbl_concat.set_index('docID')
response_tbl_concat.columns="response_"+response_tbl_concat.columns

print("response tbl before merge",len(response_tbl_concat))
response_tbl_rst=pd.merge(response_tbl_concat,download_summary[['dataset','download_status','download_data_path']],left_index=True,right_index=True,how='left')
print("response tbl after merge",len(response_tbl_rst))
# %%
print(" === save response and download document set defined ===")
response_tbl_rst=response_tbl_rst.rename(columns={'docID':'docid'})
response_tbl_rst.to_csv(PROJDIR / "data/0_metadata/dataset_2407/response_tbl_rst_2407_v1012.csv",encoding='utf-8')
response_tbl_rst.to_pickle(PROJDIR / "data/0_metadata/dataset_2407/response_tbl_rst_2407_v1012.pkl")
# %%
response_tbl_rst.dataset.isna().sum()
#.dataset.value_counts()
#.docInfoEditStatus.value_counts()
#download_summary

# %%
#DNetの場合、証券コードが4桁の場合と5桁の場合がある。5桁は末尾の1桁に0が入っている。4桁にそろえる。
# responseの条件確認
# %% 20250113 data linage fileの作成
"""descri
データリネージファイルの作成
"""
from typing import Annotated
from pydantic import BaseModel, Field,SecretStr
from pydantic.functional_validators import BeforeValidator

    
class DataLinageJson(BaseModel):
    """
    """
    create_date: str = Field(description="作成日")
    check_date: str = Field(description="チェック日")
    size: str = Field(description="ファイルサイズ")
    file_path: str = Field(description="ファイルパス")
    input_data: dict
    input_data_providing_func: dict
    index_name: str = Field(description="index")
    header: list = Field(description="header")
    count: int = Field(description="count")
    unique_count_index: int = Field(description="unique count index")
    unique_count_header: dict = Field(description="unique count")
    example_rcd: dict
    header_note: str = Field(description="header note")
    src: str = Field(description="script")
    processing: str = Field(description="processing")
    assertion: str = Field(description="assertion")
    note: str = Field(description="note")

    def save(self):
        file_name_wo_ext = Path(self.file_path).stem
        dir_name = Path(self.file_path).parent
        file_name = f"{file_name_wo_ext}.json"
        print(dir_name / file_name)
        with open(dir_name / file_name, 'w') as f:
            f.write(self.json())



header_note_txt = """
    docID: filename
    edinetCode: EDINETコード
    secCode: 証券コード
    filerName: 提出者名
    fundCode: ファンドコード
    ordinanceCode: 政令コード
    formCode: 様式コード
    docTypeCode: 書類種別コード
    periodStart: 開始期間
    periodEnd: 終了期間
    withdrawalStatus: 取下書は"1"、取り下げられた書類は"2"、それ以外は"0"が出力
    docInfoEditStatus: 財務局職員が書類を修正した情報は"1"、修正された書類は"2"、それ以外は"0"が出力
    参考: 11_EDINET_API仕様書（version 2）.pdfより
"""

DataLinageJson_obj = DataLinageJson(**{
    "create_date": "2024/10/12",
    "check_date": "2025/01/13",
    "size": f'{os.path.getsize(PROJDIR / "data/0_metadata/dataset_2407/response_tbl_rst_2407_v1012.pkl"):,}',
    "file_path": str(PROJDIR / "data/0_metadata/dataset_2407/response_tbl_rst_2407_v1012.pkl"),
    "reader": "pandas.read_pickle",
    "encoding": "utf-8",
    "input_data": {
        "response_tbl":["response_tbl_1","response_tbl_2","response_tbl_3"],
        "download_summary":["download_summary","download_summary_2020","download_summary_2407"]
        },
    "input_data_providing_func": {
        "response_tbl":"metadata_loader.get_responce",
        "download_summary":"preproc_rst_loader.download_summary"
        },
    "index_name": response_tbl_rst.index.name,
    "header": list(response_tbl_rst.columns),
    "count": len(response_tbl_rst),
    "unique_count_index": response_tbl_rst.index.nunique(),
    "unique_count_header": response_tbl_rst.describe(include='all').T['unique'].to_dict(),
    "example_rcd": response_tbl_rst.iloc[0].to_dict(),
    "header_note": header_note_txt,
    "src": "data/ds01_00_check_dataset.py",
    "assertion": assertion_text,
    "processing": processing_text,
    "note": ""
})
DataLinageJson_obj.save()

    

# %%

# %%
