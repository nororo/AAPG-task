

'''
src
dataloader
'''


# %%
#from os.path import basename as os_basename
#import re
#from traceback import format_exc
#from pandas import DataFrame as pd_DataFrame
#from pandas import to_datetime as pd_to_datetime


#from os.path import basename as os_basename
#from os.path import isfile as os_isfile
#from zipfile import ZipFile
import joblib
import glob
import pandas as pd
from tqdm import tqdm
from pathlib import Path


#from edinet_xbrl.edinet_xbrl_parser import EdinetXbrlParser
#from time import sleep

PROJPATH=r"/Users/noro/Documents/Projects/XBRL_common_space_projection/"
import numpy as np


def get_docid_list_2020()->set:
    filename=PROJPATH+"data/0_metadata/dataset2003/filelist_1912.txt"
    docid1912=set(pd.read_csv(filename,sep='\t',header=None)[0])
    filename=PROJPATH+"data/0_metadata/dataset2003/filelist_1906.txt"
    docid1906=set(pd.read_csv(filename,sep='\t',header=None)[0])
    filename=PROJPATH+"data/0_metadata/dataset2003/filelist_2003.txt"
    docid2003=set(pd.read_csv(filename,sep='\t',header=None)[0])
    
    filename=PROJPATH+"data/0_metadata/dataset2003/filelist_ex2014.txt"
    docid2014ex=set(pd.read_csv(filename,sep='\t',header=None)[0])
    filename=PROJPATH+"data/0_metadata/dataset2003/filelist_ex2015.txt"
    docid2015ex=set(pd.read_csv(filename,sep='\t',header=None)[0])
    filename=PROJPATH+"data/0_metadata/dataset2003/filelist_ex2016.txt"
    docid2016ex=set(pd.read_csv(filename,sep='\t',header=None)[0])
    filename=PROJPATH+"data/0_metadata/dataset2003/filelist_ex2017.txt"
    docid2017ex=set(pd.read_csv(filename,sep='\t',header=None)[0])
    filename=PROJPATH+"data/0_metadata/dataset2003/filelist_ex2018.txt"
    docid2018ex=set(pd.read_csv(filename,sep='\t',header=None)[0])
    filename=PROJPATH+"data/0_metadata/dataset2003/filelist_ex2019.txt"
    docid2019ex=set(pd.read_csv(filename,sep='\t',header=None)[0])
    filename=PROJPATH+"data/0_metadata/dataset2003/filelist_ex2020.txt"
    docid2020ex=set(pd.read_csv(filename,sep='\t',header=None)[0])

    return (docid2003|docid1906|docid1912|docid2014ex|docid2015ex|docid2016ex|docid2017ex|docid2018ex|docid2019ex|docid2020ex)
#    return (docid2003|docid1906|docid1912)

def get_responce(dataset_year:str='2020')->pd.DataFrame:
    
    columns_type_dict={
        'response_seqNumber':int, # 同日に提出された書類に提出時間順につく番号 YYYY/MM/DD-senCumberが提出順序情報になる
        'response_docID':str, # filename
        'response_edinetCode':str, # EDINETコード
        'response_secCode':str, # 証券コード
        'response_JCN':str, # 法人番号
        'response_filerName':str, # 提出者名
        'response_fundCode':str, # ファンドコード
        'response_ordinanceCode':str, # 政令コード
        'response_formCode':str, # 様式コード
        'response_docTypeCode':str, # 書類種別コード
        'response_periodStart':str, # 開始期間
        'response_periodEnd':str, # 終了期間
        'response_submitDateTime':str, # 書類提出日時 
        'response_docDescription':str, # EDINET の閲覧サイトの書類検索結 果画面において、「提出書類」欄に表 示される文字列
        'response_issuerEdinetCode':str, # 発行会社EDINETコード 大量保有について発行会社の EDINETコード
        'response_subjectEdinetCode':str, # 公開買付けについて対象となる EDINETコード
        'response_subsidiaryEdinetCode':str, # 子会社の EDINET コードが出力され ます。複数存在する場合(最大 10 個)、","(カンマ)で結合した文字列 が出力されます
        'response_currentReportReason':str, # 臨報提出事由 臨時報告書の提出事由が出力され ます。複数存在する場合、","(カン マ)で結合した文字列が出力されます。
        'response_parentDocID':str, # 親書類管理番号
        'response_opeDateTime':str, # 「2-1-6 財務局職員による書類情報 修正」、「2-1-7 財務局職員による書 類の不開示」、磁気ディスク提出及 び紙面提出を行った日時が出力されます
        'response_withdrawalStatus':int, # 取下書は"1"、取り下げられた書類 は"2"、それ以外は"0"が出力されます
        'response_docInfoEditStatus':int, # 財務局職員が書類を修正した情報 は"1"、修正された書類は"2"、それ 以外は"0"が出力されます
        'response_disclosureStatus':int, # 財務局職員によって書類の不開示を 開始した情報は"1"、不開示とされて いる書類は"2"、財務局職員によっ て書類の不開示を解除した情報は "3"、それ以外は"0"が出力されます。
        'response_xbrlFlag':int,# 書類に XBRL がある場合は"1" それ以外0
        'response_pdfFlag':int,# 書類に PDF がある場合は"1" それ以外0
        'response_attachDocFlag':int, # 書類に代替書面・添付文書がある場合:1 それ以外:0
        'response_englishDocFlag':int, # 書類に英文ファイルがある場合1
        'response_csvFlag':int, # 書類にcsvがある場合1
        'response_legalStatus':int, # "1":縦覧中 "2":延長期間中(法定縦覧期間満 了書類だが引き続き閲覧可能。) "0":閲覧期間満了(縦覧期間満了 かつ延長期間なし、延長期間満了 又は取下げにより閲覧できないも の。なお、不開示は含まない。)

        'parse_security_code':str
    }
    if dataset_year=='2020':
        # -1803
        filename=PROJPATH+"data/0_metadata/dataset2003/EDINET_response.csv"
        response_data1803=pd.read_csv(filename,encoding='utf-8',index_col=0,dtype=columns_type_dict).reset_index(drop=True)

        # 1903
        filename=PROJPATH+"data/0_metadata/dataset2003/EDINET_response_1903.csv"
        response_data1903=pd.read_csv(filename,encoding='utf-8',index_col=0,dtype=columns_type_dict).reset_index(drop=True)

        edinet_res_file=PROJPATH+'data/0_metadata/dataset2003/EDINET_response_0621.csv'
        response_data1903_0621=pd.read_csv(edinet_res_file,encoding='utf-8',index_col=0,dtype=columns_type_dict).reset_index(drop=True)
        # 2020
        edinet_res_file=PROJPATH+'data/0_metadata/dataset2003/EDINET_response_20200904.csv'
        response_data_20200904=pd.read_csv(edinet_res_file,encoding='utf-8',index_col=0,dtype=columns_type_dict).reset_index(drop=True)

        response_tbl=pd.concat([response_data1803,response_data1903,response_data1903_0621,response_data_20200904],sort=False).reset_index(drop=True)
        response_tbl=response_tbl.assign(opeDateTime_ts=pd.to_datetime(response_tbl.opeDateTime))

        response_tbl.secCode=float_to_str(response_tbl.secCode)
        response_tbl.JCN=float_to_str(response_tbl.JCN)
        response_tbl.ordinanceCode=float_to_str(response_tbl.ordinanceCode)
        response_tbl.docTypeCode=float_to_str(response_tbl.docTypeCode)

    if dataset_year=='2023_org':
        filename=PROJPATH+"data/0_metadata/trial/responce_meta.pkl.cmp"
        results_1 = joblib.load(filename)
        filename=PROJPATH+"data/0_metadata/trial/responce_meta_retry.pkl.cmp"
        results_2 = joblib.load(filename)
        results_all=results_1+results_2
        data_list = [item['data'] for item in results_all if item['status'] == 'Success']
        response_tbl=pd.concat(data_list).reset_index()

        response_tbl.secCode=float_to_str(response_tbl.secCode)
        response_tbl.JCN=float_to_str(response_tbl.JCN)
        response_tbl.ordinanceCode=float_to_str(response_tbl.ordinanceCode)
        response_tbl.docTypeCode=float_to_str(response_tbl.docTypeCode)

    if dataset_year=='2023':
        filename=PROJPATH+"data/0_metadata/trial/response_meta_convert.pkl.cmp"
        response_tbl=pd.read_pickle(filename)

    if dataset_year=='2407':
        filename=PROJPATH+"data/0_metadata/trial/responce_meta_20230901_20240722.pkl.cmp"
        results_1 = joblib.load(filename)
        data_list = [item['data'] for item in results_1 if item['status'] == 'Success']
        response_tbl=pd.concat(data_list).reset_index()

    #assert response_tbl.query("docTypeCode=='120' and docInfoEditStatus!='2'").docID.duplicated().sum()==0
    return response_tbl


def float_to_str(series):
    return series.astype('str').str.replace('\.0','').replace('nan', np.nan)

def get_edinetcode(dataset_year:str='2020')->set:
    if dataset_year=='2020':
        edinetcode=pd.read_csv(PROJPATH+'data/0_metadata/dataset2003/EdinetcodeDlInfo1903.csv', \
                            header=1,index_col=False, engine="python", encoding="cp932")
        edinetcode2=pd.read_csv(PROJPATH+'data/0_metadata/dataset2003/EdinetcodeDlInfo.csv', \
                            header=1,index_col=False, engine="python", encoding="cp932")
        code_2020=set(edinetcode.query("上場区分=='上場'")['ＥＤＩＮＥＴコード'])
        code_2020_2=set(edinetcode2.query("上場区分=='上場'")['ＥＤＩＮＥＴコード'])
        edinetcodelist=code_2020|code_2020_2
    if dataset_year=='2023and1903':
        edinetcode2023=pd.read_csv(PROJPATH+'data/0_metadata/trial/EdinetcodeDlInfo2312.csv', \
                            header=1,index_col=False, engine="python", encoding="cp932")
        edinetcode=pd.read_csv(PROJPATH+'data/0_metadata/dataset2003/EdinetcodeDlInfo1903.csv', \
                            header=1,index_col=False, engine="python", encoding="cp932")
        
        code_2023=set(edinetcode2023.query("上場区分=='上場'")['ＥＤＩＮＥＴコード'])
        code_2020=set(edinetcode.query("上場区分=='上場'")['ＥＤＩＮＥＴコード'])
        edinetcodelist=code_2023|code_2020
    if dataset_year=='2023':
        edinetcode2023=pd.read_csv(PROJPATH+'data/0_metadata/trial/EdinetcodeDlInfo2312.csv', \
                            header=1,index_col=False, engine="python", encoding="cp932")
        
        edinetcodelist=set(edinetcode2023.query("上場区分=='上場'")['ＥＤＩＮＥＴコード'])
    
    if dataset_year=='2407':
        edinetcode2407=pd.read_csv(PROJPATH+'data/0_metadata/trial/EdinetcodeDlInfo2407.csv', \
                            header=1,index_col=False, engine="python", encoding="cp932")
        
        edinetcodelist=set(edinetcode2407.query("上場区分=='上場'")['ＥＤＩＮＥＴコード'])
    
        
    return edinetcodelist


class bs_pl_table():
    """
    type
        all columns other than 'depth': string
        depth: float
    """
    def __init__(self):
        self.f_name_dict={
            2014:'2014account_list.xls',
            2015:'2015account_list.xls',
            2016:'2016account_list.xls',
            2017:'2017account_list.xls',
            2018:'1f_AccountList_2018.xls',
            2019:'1f_AccountList_2019.xls',
            2020:'1f_AccountList_2020.xlsx',
            2021:'1f_AccountList_2021.xlsx',
            2022:'1f_AccountList_2022.xlsx',
            2023:'1f_AccountList_2023.xlsx',
            2024:'1f_AccountList_2024.xlsx'
        }
        self.rename_dict={
            '科目分類':'account_type',
            '標準ラベル（日本語）':'label_jp',
            '冗長ラベル（日本語）':'label_jp_long',
            '標準ラベル（英語）':'label_en',
            '冗長ラベル（英語）':'label_en_long',
            '用途区分、財務諸表区分及び業種区分のラベル（日本語）':'label_jp_purpose',
            '用途区分、財務諸表区分及び業種区分のラベル（英語）':'label_en_purpose',
            '名前空間プレフィックス':'namespace_prefix',
            '要素名':'element_name',
        }
        self.out_columns=[
            'account_type', 'label_jp', 'label_jp_long', 'label_en', 'label_en_long',
            'label_jp_purpose', 'label_en_purpose', 'namespace_prefix', 'element_name',
            #'type',
            #'substitutionGroup',
            'periodType',
            'balance', 'abstract', 'depth', 'key_cap', 'parent_key','key','schima','taxonomi',
            'is_parent_abstruct', 'sum_flg', 'bussiness_type_num_str', 'year_str','tbl_name',
            #'参照リンク'
            ]

#        self.out_columns=['科目分類', '標準ラベル（日本語）', '冗長ラベル（日本語）', '標準ラベル（英語）', '冗長ラベル（英語）',
#            '用途区分、財務諸表区分及び業種区分のラベル（日本語）', '用途区分、財務諸表区分及び業種区分のラベル（英語）',
#            '名前空間プレフィックス', '要素名', 'type', 'substitutionGroup', 'periodType',
#            'balance', 'abstract', 'depth', 'key_cap', 'parent_key','key','schima','taxonomi',
#            'is_parent_abstruct', 'sum_flg', 'bussiness_type_num_str', 'year_str','tbl_name',
#            #'参照リンク'
#            ]
        self.integlated_list_path=Path("/Users/noro/Documents/Projects/XBRL_common_space_projection/data/2_intermediate/account_list")
        if self.integlated_list_path.is_dir():
            self.tbl_bs_c=self.post_proc(pd.read_csv(str(self.integlated_list_path / "account_list_bs.csv"),encoding='shift-jis',dtype=str,index_col=None))
            self.tbl_bs_c.depth=self.tbl_bs_c.depth.astype(float)

            self.tbl_pl_c=self.post_proc(pd.read_csv(str(self.integlated_list_path / "account_list_pl.csv"),encoding='shift-jis',dtype=str,index_col=None))
            self.tbl_pl_c.depth=self.tbl_pl_c.depth.astype(float)
            self.tbl_oci_c=self.post_proc(pd.read_csv(str(self.integlated_list_path / "account_list_oci.csv"),encoding='shift-jis',dtype=str,index_col=None))
            self.tbl_oci_c.depth=self.tbl_oci_c.depth.astype(float)
            self.tbl_ss_c=self.post_proc(pd.read_csv(str(self.integlated_list_path / "account_list_ss.csv"),encoding='shift-jis',dtype=str,index_col=None))
            self.tbl_ss_c.depth=self.tbl_ss_c.depth.astype(float)
            self.tbl_cf_c=self.post_proc(pd.read_csv(str(self.integlated_list_path / "account_list_cf.csv"),encoding='shift-jis',dtype=str,index_col=None))
            self.tbl_cf_c.depth=self.tbl_cf_c.depth.astype(float)
            self.load=True
        else:
            self.load=False
    def post_proc(self,tbl):
        tbl.depth=tbl.depth.astype(float)
        tbl.balance=tbl.balance.fillna('-')
        return tbl.rename(columns=self.rename_dict)
    def get_bs_pl_all(self):
        if self.load:
            return self.tbl_bs_c[self.out_columns],self.tbl_pl_c[self.out_columns],self.tbl_oci_c[self.out_columns],self.tbl_ss_c[self.out_columns],self.tbl_cf_c[self.out_columns]
        else:
            return self.make_bs_pl_all()

    def make_bs_pl_all(self):
        #filename="/Users/noro/Documents/Projects/XBRLanalysis/data/bkup/metadata/"+str(year)+"account_list.xls"
        tbl_bs_c=pd.DataFrame()
        tbl_pl_c=pd.DataFrame()
        tbl_oci_c=pd.DataFrame()
        tbl_ss_c=pd.DataFrame()
        tbl_cf_c=pd.DataFrame()

        for year in self.f_name_dict.keys():
            filename="/Users/noro/Documents/Projects/XBRL_common_space_projection/data/0_metadata/xbrl_keys/"+self.f_name_dict[year]
            print(filename)
            #filename="/Users/noro/Documents/Projects/XBRLanalysis/data/bkup/metadata/"+self.f_name_dict[year]
            if year>=2020:
                book=pd.ExcelFile(filename,engine="openpyxl")
            else:
                book=pd.ExcelFile(filename)

            for itr in range(2,len(book.sheet_names)-1):
                #self.itr=itr
                #self.fname=self.f_name_dict[year]
                sheet_name=itr
                if itr ==2:
                    tbl_bs,tbl_pl,tbl_oci,tbl_ss,tbl_cf=self.get_bs_pl(book,sheet_name,year)
                    tbl_bs_c=pd.concat([tbl_bs_c,tbl_bs],axis=0)
                    tbl_pl_c=pd.concat([tbl_pl_c,tbl_pl],axis=0)
                    tbl_oci_c=pd.concat([tbl_oci_c,tbl_oci],axis=0)
                    tbl_ss_c=pd.concat([tbl_ss_c,tbl_ss],axis=0)
                    tbl_cf_c=pd.concat([tbl_cf_c,tbl_cf],axis=0)

                else:
                    tbl_bs,tbl_pl,tbl_ss,tbl_cf=self.get_bs_pl(book,sheet_name,year)
                    tbl_bs2,tbl_pl2,tbl_oci,tbl_ss2,tbl_cf2=self.get_bs_pl(book,2,year)
                    tbl_bs=self.merge_general_mst(tbl_bs,tbl_bs2)
                    tbl_pl=self.merge_general_mst(tbl_pl,tbl_pl2)
                    tbl_ss=self.merge_general_mst(tbl_ss,tbl_ss2)
                    #tbl_oci=self.merge_general_mst(tbl_oci,tbl_oci2)
                    tbl_cf=self.merge_general_mst(tbl_cf,tbl_cf2)

                    tbl_bs_c=pd.concat([tbl_bs_c,tbl_bs],axis=0)
                    tbl_pl_c=pd.concat([tbl_pl_c,tbl_pl],axis=0)
                    tbl_ss_c=pd.concat([tbl_ss_c,tbl_ss],axis=0)
                    tbl_cf_c=pd.concat([tbl_cf_c,tbl_cf],axis=0)
                    
                self.tbl_bs_c=tbl_bs_c[self.out_columns].rename(columns=self.rename_dict)
                self.tbl_pl_c=tbl_pl_c[self.out_columns].rename(columns=self.rename_dict)
                self.tbl_oci_c=tbl_oci_c[self.out_columns].rename(columns=self.rename_dict)
                self.tbl_ss_c=tbl_ss_c[self.out_columns].rename(columns=self.rename_dict)
                self.tbl_cf_c=tbl_cf_c[self.out_columns].rename(columns=self.rename_dict)
                self.save_bs_pl()
        return self.tbl_bs_c[self.out_columns],self.tbl_pl_c[self.out_columns],self.tbl_oci_c[self.out_columns],self.tbl_ss_c[self.out_columns],self.tbl_cf_c[self.out_columns]

    def save_bs_pl(self):
        self.integlated_list_path.mkdir(parents=True,exist_ok=True)
        self.tbl_bs_c.to_csv(str(self.integlated_list_path / "account_list_bs.csv"),encoding='shift-jis')
        self.tbl_pl_c.to_csv(str(self.integlated_list_path / "account_list_pl.csv"),encoding='shift-jis')
        self.tbl_oci_c.to_csv(str(self.integlated_list_path / "account_list_oci.csv"),encoding='shift-jis')
        self.tbl_ss_c.to_csv(str(self.integlated_list_path / "account_list_ss.csv"),encoding='shift-jis')
        self.tbl_cf_c.to_csv(str(self.integlated_list_path / "account_list_cf.csv"),encoding='shift-jis')
    
    def get_bs_pl(self,book,sheet_name,year):
        sheet=book.parse(sheet_name=sheet_name,header=1,index_col=0).reset_index()
        mask=sheet['科目分類'].str.extract('(.+科目一覧)').notna()
        startpoint=np.where(mask)[0]
        tbl_bs=sheet.iloc[:startpoint[0]-1].reset_index(drop=True)
        tbl_bs=self._prep(tbl_bs,year,sheet_name)
        tbl_bs=tbl_bs.assign(tbl_name='BS')

        tbl_pl=sheet.iloc[startpoint[0]+2:startpoint[1]-1].reset_index(drop=True)
        tbl_pl=self._prep(tbl_pl,year,sheet_name)
        tbl_pl=tbl_pl.assign(tbl_name='PL')

        if sheet_name==2:
            tbl_oci=sheet.iloc[startpoint[1]+2:startpoint[2]-1].reset_index(drop=True)
            tbl_oci=self._prep(tbl_oci,year,sheet_name)
            tbl_oci=tbl_oci.assign(tbl_name='OCI')
            tbl_ss=sheet.iloc[startpoint[2]+2:startpoint[3]-1].reset_index(drop=True)
            tbl_ss=self._prep(tbl_ss,year,sheet_name)
            tbl_ss=tbl_ss.assign(tbl_name='SS')
            tbl_cf=sheet.iloc[startpoint[3]+2:].reset_index(drop=True)
            mask_cf=tbl_cf['科目分類'].notna()
            tbl_cf=tbl_cf.loc[mask_cf,:]
            tbl_cf=self._prep(tbl_cf,year,sheet_name)
            tbl_cf=tbl_cf.assign(tbl_name='CF')
    
            return tbl_bs,tbl_pl,tbl_oci,tbl_ss,tbl_cf
        else:
            tbl_ss=sheet.iloc[startpoint[1]+2:startpoint[2]-1].reset_index(drop=True)
            tbl_ss=self._prep(tbl_ss,year,sheet_name)
            tbl_ss=tbl_ss.assign(tbl_name='SS')
            tbl_cf=sheet.iloc[startpoint[2]+2:].reset_index(drop=True)
            mask_cf=tbl_cf['科目分類'].notna()
            tbl_cf=tbl_cf.loc[mask_cf,:]
            tbl_cf=self._prep(tbl_cf,year,sheet_name)
            tbl_cf=tbl_cf.assign(tbl_name='CF')
            return tbl_bs,tbl_pl,tbl_ss,tbl_cf

    def _prep(self,tbl,year,sheet_name):
        tbl=tbl.assign(key_cap=tbl['名前空間プレフィックス']+":"+tbl['要素名'])
        tbl=tbl.assign(
            key=tbl.key_cap.str.lower(),
            parent_key=['no_parent_key']+[ self._get_parent_key(tbl,itr) for itr in range(1,len(tbl))],
            is_parent_abstruct=[True]+[ self._is_parent_abstruct(tbl,itr) for itr in range(1,len(tbl))],
            # 日本語のカラムだと、ラベル名称に「合計」が入ってる場合がある。
            sum_flg=((tbl['用途区分、財務諸表区分及び業種区分のラベル（英語）'].str.extract('(.+合計)').notna())*1).astype(int).astype(str),
            bussiness_type_num_str=str(sheet_name),
            year_str=str(year)
            )
        tbl['schima']=tbl.key.str.split(':',expand=True)[0]
        tbl['taxonomi']=(tbl.key.str.split(':',expand=True)[1]).fillna('-')
        tbl.depth=tbl.depth.astype(float)
        return tbl

    def _get_parent_key(self,tbl_bs,itr):
        #self.tbl_bs=tbl_bs
        #self.itr=itr
        serch_obj=tbl_bs.depth.iloc[itr]-1
        rcd_obj=tbl_bs.query("depth==@serch_obj and index < @itr").tail(1)
        
        return rcd_obj.key_cap.values[0]

    def _is_parent_abstruct(self,tbl_bs,itr):
        serch_obj=tbl_bs.depth.iloc[itr]-1
        rcd_obj=tbl_bs.query("depth==@serch_obj and index < @itr").tail(1)
        return rcd_obj.abstract.values[0]=='true'
    
    def merge_general_mst(self,tbl_bs,tbl_bs2):
        # TODO otherwize than BNK should be considered
        #if tbl_bs.bussiness_type_num_str_base.max()=='4':
        tbl_bs=tbl_bs.assign(key_cap_normalized=tbl_bs.key_cap
                             #.str.removesuffix('AssetsBNK')
                             #.str.removesuffix('LiabilitiesBNK')
                             #.str.removesuffix('OIBNK')
                             .str.removesuffix('BNK')
                             .str.removesuffix('CNS')
                             .str.removesuffix('CNA')
                             .str.removesuffix('SEC')
                             .str.removesuffix('INS')
                             .str.removesuffix('RWY')
                             .str.removesuffix('WAT')
                             .str.removesuffix('NWY')
                             .str.removesuffix('telecommunications')
                             .str.removesuffix('ELE')
                             .str.removesuffix('GAS')
                             .str.removesuffix('LIQ')
                             .str.removesuffix('IVT')
                             .str.removesuffix('INV')
                             .str.removesuffix('SPF')
                             .str.removesuffix('MED')
                             .str.removesuffix('EDU')
                             .str.removesuffix('CMD')
                             .str.removesuffix('LEA')
                             .str.removesuffix('FND')
        )
        #tbl_bs2=tbl_bs2.assign(key_cap_normalized=tbl_bs2.key_cap
        #                       .str.removesuffix('CA')
        #                       .str.removesuffix('OA')
        #                       )
        tbl_bs2=tbl_bs2.assign(match_flg=(tbl_bs2.key_cap.isin(tbl_bs.key_cap_normalized)).astype(int))
        tbl_bs2_match=pd.merge(tbl_bs2.query("match_flg==1"),
                    tbl_bs[['key_cap_normalized','depth','parent_key','is_parent_abstruct','bussiness_type_num_str']].rename(columns={'depth':'depth_base','parent_key':'parent_key_base','is_parent_abstruct':'is_parent_abstruct_base','bussiness_type_num_str':'bussiness_type_num_str_base'}),left_on='key_cap',right_on='key_cap_normalized',how='left')
        # matchした場合、depthとparentを変更
        tbl_bs2_match=tbl_bs2_match.assign(depth=tbl_bs2_match.depth_base,
                    parent_key=tbl_bs2_match.parent_key_base,
                    is_parent_abstruct=tbl_bs2_match.is_parent_abstruct_base,
                    key_cap=tbl_bs2_match.key_cap,
                    bussiness_type_num_str=tbl_bs2_match.bussiness_type_num_str_base,
                    )
        # unmatchedのaccountのdepthをparentに基づいて変更 depth++1の繰り返し処理
        max_depth=int(tbl_bs2.depth.max())
        for itr_depth in range(2,max_depth+1):
            tbl_bs2_match_add=pd.merge(tbl_bs2.query("depth==@itr_depth and match_flg==0"),tbl_bs2_match[['key_cap_normalized','depth_base','bussiness_type_num_str_base']],left_on='parent_key',right_on='key_cap_normalized',how='left')
            tbl_bs2_match_add=tbl_bs2_match_add.assign(depth=tbl_bs2_match_add.depth_base+1,
                                    bussiness_type_num_str=tbl_bs2_match_add.bussiness_type_num_str_base
                                    )
            tbl_bs2_match_add=tbl_bs2_match_add.assign(depth_base=tbl_bs2_match_add.depth,
                                    key_cap_normalized=tbl_bs2_match_add.key_cap
                                    )
            tbl_bs2_match=pd.concat([tbl_bs2_match,tbl_bs2_match_add],axis=0)
        return pd.concat([tbl_bs,tbl_bs2_match[self.out_columns]],axis=0).drop_duplicates(keep='last',subset='key_cap')
