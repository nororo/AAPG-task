
"""
Extract KAM from downloaded XBRL files from EDINET API2

"""

# %%



import pandas as pd
import numpy as np
from pathlib import Path
import sys


from pathlib import Path

import unicodedata
import string
    

import json
import joblib
import os
from tqdm import tqdm
from time import sleep
import datetime
import xml.etree.ElementTree as ET
import re
from zipfile import ZipFile

import warnings
warnings.filterwarnings('ignore')
import pandera as pa
from pandera.typing import DataFrame, Series

PROJPATH=r"PROJECT_PATH"
PROJDIR=Path(PROJPATH)


from arelle import Cntlr
class xbrl_elm_schima(pa.DataFrameModel):
    """
        key:prefix+":"+element_name
        data_str
        context_ref
    """
    key: Series[str] = pa.Field(nullable=True)
    data_str: Series[str] = pa.Field(nullable=True)
    context_ref: Series[str] = pa.Field(nullable=True)
    decimals: Series[str] = pa.Field(nullable=True)# T:-3, M:-6, B:-9
    precision: Series[str] = pa.Field(nullable=True)
    element_name: Series[str] = pa.Field(nullable=True)
    unit: Series[str] = pa.Field(nullable=True)# 'JPY'
    period_type: Series[str] = pa.Field(isin=['instant','duration'],nullable=True) # 'instant','duration'
    isTextBlock_flg: Series[int] = pa.Field(isin=[0,1],nullable=True) # 0,1
    abstract_flg: Series[int] = pa.Field(isin=[0,1],nullable=True) # 0,1
    period_start: Series[str] = pa.Field(nullable=True)
    period_end: Series[str] = pa.Field(nullable=True)
    instant_date: Series[str] = pa.Field(nullable=True)
    
def get_fact_data(fact)->dict:
    fact_data = {
        'key':str(fact.qname),
        'data_str':fact.value,
        'decimals':fact.decimals,
        'precision':fact.precision,
        'context_ref':fact.contextID,
        'element_name':str(fact.qname.localName),
        'unit':fact.unitID,#(str) – unitRef attribute
        'period_type':fact.concept.periodType,#'instant','duration'
        'isTextBlock_flg':int(fact.concept.isTextBlock), # 0,1
        'abstract_flg':int(fact.concept.abstract=='true'), # Note: fatc.concept.abstract is str not bool.
    }
    if fact.context.startDatetime:
        fact_data['period_start'] = fact.context.startDatetime.strftime('%Y-%m-%d'),
    else:
        fact_data['period_start'] = None
    if fact.context.endDatetime:
        fact_data['period_end'] = fact.context.endDatetime.strftime('%Y-%m-%d'), # 1 day added???
    else:
        fact_data['period_end'] = None
    if fact.context.instantDatetime:
        fact_data['instant_date'] = fact.context.instantDatetime.strftime('%Y-%m-%d'), # 1 day added???
    else:
        fact_data['instant_date'] = None

       
    fact_data['end_date_pv']=None
    fact_data['instant_date_pv']=None
    for item in fact.context.propertyView:
                if item:
                    if item[0] == 'endDate':
                        fact_data['end_date_pv'] = item[1]
                    elif item[0] == 'instant':
                        fact_data['instant_date_pv'] = item[1]
    scenario = []
    for (dimension, dim_value) in fact.context.scenDimValues.items():
        scenario.append({
            'ja': (
                dimension.label(preferredLabel=None, lang='ja', linkroleHint=None),
                dim_value.member.label(preferredLabel=None, lang='ja', linkroleHint=None)),
            'en': (
                dimension.label(preferredLabel=None, lang='en', linkroleHint=None),
                dim_value.member.label(preferredLabel=None, lang='en', linkroleHint=None)),
            'id': (
                dimension.id,
                dim_value.member.id),
        })
    if scenario:
            scenario_json = json.dumps(
                scenario, ensure_ascii=False, separators=(',', ':'))
    else:
        scenario_json = None

    fact_data['scenario'] = scenario_json
    return fact_data

# %%

def get_xbrl_df(xbrl_filename:str,log_dict)->xbrl_elm_schima:
    """
    v0820
    arelle.ModelInstanceObject - Arelle
        https://arelle.readthedocs.io/en/2.18.0/apidocs/arelle/arelle.ModelInstanceObject.html#arelle.ModelInstanceObject.ModelFact
    
    """
    if log_dict['arelle_log_fname'] is None:
        log_dict['arelle_log_fname'] = str(TESTDIR / "arelle.log")

    ctrl = Cntlr.Cntlr(logFileName=str(log_dict['arelle_log_fname']))
    model_xbrl = ctrl.modelManager.load(xbrl_filename)
    if len(model_xbrl.facts)==0:
        log_dict['xbrl_load_status']="Failure"
        ctrl.close()
        return pd.DataFrame(columns=get_columns(xbrl_elm_schima)),log_dict
    else:
        log_dict['xbrl_load_status']="Success"
        fact_dict_list = []
        for fact in model_xbrl.facts:
            fact_dict_list.append(get_fact_data(fact))
        # log
        ctrl.close()
        return pd.DataFrame(fact_dict_list).drop_duplicates(),log_dict


def preproc_nlp_kam(text:str)->str:
    # unicode
    replaced_text = unicodedata.normalize("NFKC", text)
    # drop number
    #replaced_text = drop_number(replaced_text)
    # drop signature 1
    #replaced_text = re.sub(re.compile("[!-/:-@[-`{-~]"), '', replaced_text)
    # drop signature 2
    #replaced_text = re.sub(r'\(', '', replaced_text)
    
    # drop signature 3
    #table = str.maketrans("", "", string.punctuation  + "◆■※【】)(「」、。・")
    #replaced_text = replaced_text.translate(table)
    # drop return (recursive)
    replaced_text=data_utils.RtnDroper(replaced_text)    
    
    return replaced_text

def preproc_nlp_kam_aud(text:str)->str:
    # unicode
    replaced_text = unicodedata.normalize("NFKC", text)
    # drop number
    #replaced_text = drop_number(replaced_text)
    # drop signature 1
    #replaced_text = re.sub(re.compile("[!-/:-@[-`{-~]"), '', replaced_text)
    # drop signature 2
    replaced_text = re.sub('当監査法人', '監査人', replaced_text)
    replaced_text = re.sub('した。', 'する。', replaced_text)
    replaced_text = re.sub('当てた。', '当てる。', replaced_text)
    replaced_text = re.sub('行った。', '行う。', replaced_text)
    replaced_text = re.sub('確かめた。', '確かめる。', replaced_text)
    
    # drop signature 3
    #table = str.maketrans("", "", string.punctuation  + "◆■※【】)(「」、。・")
    #replaced_text = replaced_text.translate(table)
    # drop return (recursive)
    replaced_text=data_utils.RtnDroper(replaced_text)    
    
    return replaced_text

PROJDIR=Path(PROJPATH)
# %%

class p_edges_schima(pa.DataFrameModel):
    """
    KEY:
        parent: jppfs_cor_CameCaseAccountName
        child: jppfs_cor_CameCaseAccountName
        
        parent_taxonomi_tag: jppfs_cor_accountname
        child_taxonomi_tag: jppfs_cor_accountname
        
            taxonomi_tag <- locators_df.schima_taxonomi.str.lower()
            schima_taxonomi <- attr_sr[attr_sr.index.str.contains('href')].values[0].split('#')[1]
    """
    parent_key: Series[str]
    child_key: Series[str]
    role: Series[str]
    child_order: Series[str]

class original_account_list_schima(pa.DataFrameModel):
    """
        label: 
        key: jpcrp030000-asr_E37207-000:IncreaseDecreaseInIncomeTaxesPayableOpeCF
        role: 
        (schima_taxonomi: schima_taxonomi like)
            jpcrp030000-asr_E37207-000_IncreaseDecreaseInIncomeTaxesPayableOpeCF
            sepalated it by '#' and get later part that is jpcrp030000-asr-001_E37207-000_2023-06-30_01_2023-09-29.xsd#jpcrp030000-asr_E37207-000_IncreaseDecreaseInIncomeTaxesPayableOpeCF
            (from xlink:href in pre.xml file) 
    """
    #schima_taxonomi: Series[str]
    label: Series[str]
    key: Series[str]
    role: Series[str]

# %%


def dtype(df_type, use_nullable=True):
    """
    REF
    https://qiita.com/SaitoTsutomu/items/ce632ac852f8b72b56db
    """
    dc = {}
    schema = df_type.to_schema()
    for name, column in schema.columns.items():
        typ = column.dtype.type
        if use_nullable and column.nullable and column.dtype.type == int:
            typ = "Int64"
        dc[name] = typ
    return dc

def get_xbrl_rapper_aud(docid,out_path,update_flg=False):
    log_dict = {"is_xbrl_file":None, "is_xsd_file":None, "arelle_log_fname":None,"status":None,"error_message":None}
    try:
        exist_flg=(out_path / "audit_xbrl_proc_pd222.csv").exists()
        if (exist_flg)&(update_flg==False):
            xbrl_processed=pd.read_csv(out_path / "audit_xbrl_proc_pd222.csv",dtype=dtype(xbrl_elm_schima))
            return  xbrl_elm_schima(xbrl_processed),log_dict
    except Exception as e:
        exist_flg=False
        pass
    if (~exist_flg)|(update_flg==True):
        try:
            data_dir_raw=PROJDIR / "data" / "1_raw"
            zip_file = list(data_dir_raw.glob("data_pool_*/"+docid+".zip"))[0]
            with ZipFile(str(zip_file)) as zf:

                fn=[item for item in zf.namelist() if (".xbrl" in item)&("AuditDoc" in item)&("aai" in item)]
                if len(fn)>0:
                    zf.extract(fn[0], out_path)
                    log_dict["is_xbrl_file"] = True
                else:
                    log_dict["is_xbrl_file"] = False
                fn=[item for item in zf.namelist() if (".xsd" in item)&("AuditDoc" in item)&("aai" in item)]
                if len(fn)>0:
                    zf.extract(fn[0], out_path)
                    log_dict["is_xsd_file"] = True
                else:
                    log_dict["is_xsd_file"] = False
            xbrl_path=out_path / "XBRL" / "AuditDoc"
            if (len(list(xbrl_path.glob("*.xbrl")))>0)&(len(list(xbrl_path.glob("*.xsd")))>0): # xbrl and xsd file exists
                xbrl_filename = str(list(xbrl_path.glob("*.xbrl"))[0])
                (xbrl_path / "arelle.log").touch()
                log_dict["arelle_log_fname"]=str(xbrl_path / "arelle.log")
                xbrl_processed, log_dict = get_xbrl_df(xbrl_filename,log_dict)
                xbrl_processed.to_csv(out_path / "audit_xbrl_proc_pd222.csv",index=False)
                log_dict["status"] = "Success"
                return xbrl_processed,log_dict
            else:
                log_dict["status"] = "Failure"
                log_dict["error_message"] = "No xbrl or xsd file"
                return pd.DataFrame(columns=get_columns(xbrl_elm_schima)),log_dict
        except Exception as e:
            log_dict["status"] = "Failure"
            log_dict["error_message"] = e
            return pd.DataFrame(columns=get_columns(xbrl_elm_schima)),log_dict


# %%
def format_taxonomi(taxonomi_str:str)->str:
    """
    Convert
        From:
        jpcrp030000-asr_E37207-000_IncreaseDecreaseInIncomeTaxesPayableOpeCF
        To:
        jpcrp030000-asr_E37207-000:IncreaseDecreaseInIncomeTaxesPayableOpeCF
    """
    return "_".join(taxonomi_str.split('_')[:-1])+":"+taxonomi_str.split('_')[-1]

def get_columns(schima:pa.DataFrameModel)->list:
    return list(schima.to_schema().columns.keys())

def get_presentation_account_list_aud(docid:str,identifier:str,out_path)->(p_edges_schima,original_account_list_schima,dict,dict):
    """
    locator:
        (role:)
        href:
        label:
    arc:
        (role:)
        from:
        to:
        order:
        role is given to edge
    """
    dict_t={
        'docID':docid,
        'org_taxonomi_cnt':None,
        'org_taxonomi_list':[],
        'status':None,
        'error_message':None
            }
    try:
        data_dir_raw = PROJDIR / "data" / "1_raw"
        zip_file = list(data_dir_raw.glob("data_pool_*/"+docid+".zip"))[0]
        with ZipFile(str(zip_file)) as zf:
                fn=[item for item in zf.namelist() if ("pre.xml" in item)&("aai" in item)]
                if len(fn)>0:
                    zf.extract(fn[0], out_path)
        xml_def_path=out_path / "XBRL" / "AuditDoc"
        if len(list(xml_def_path.glob("*pre.xml")))==0:
            raise Exception("No pre.xml file")
        else:
            tree = ET.parse(str(list(xml_def_path.glob("*pre.xml"))[0]))
            root = tree.getroot()
    
            locators = []
            arcs = []
            for child in root:
                attr_sr_p = pd.Series(child.attrib)
                role = attr_sr_p[attr_sr_p.index.str.contains('role')].item()
                for child_of_child in child:
                    locator = {'role':role,'schima_taxonomi':None}
                    arc = {'parent':None,'child':None,'child_order':None,'role':role}
    
                    attr_sr = pd.Series(child_of_child.attrib)
                    attr_type = attr_sr[attr_sr.index.str.contains('type')].item()
                    if attr_type=='locator':
                        locator['schima_taxonomi'] = attr_sr[attr_sr.index.str.contains('href')].item().split('#')[1]
                        locator['label'] = attr_sr[attr_sr.index.str.contains('label')].item()
                    elif attr_type=='arc':
                        arc['parent'] = attr_sr[attr_sr.index.str.contains('from')].item()
                        arc['child'] = attr_sr[attr_sr.index.str.contains('to')].item()
                        arc['child_order'] = attr_sr[attr_sr.index.str.contains('order')].item()
    
                    locators.append(locator)
                    arcs.append(arc)
    
            locators_df = pd.DataFrame(locators).dropna(subset=['schima_taxonomi'])
            locators_df = locators_df.assign(
                role=locators_df.role.str.split('/',expand=True).iloc[:,-1],
                key=locators_df.schima_taxonomi.apply(format_taxonomi)
                                           )
            label_to_taxonomi_dict = locators_df.set_index('label')['key'].to_dict()
    
            p_edges_df = pd.DataFrame(arcs).dropna(subset=['child'])
            p_edges_df = p_edges_df.assign(
                parent_key=p_edges_df.parent.replace(label_to_taxonomi_dict),
                child_key = p_edges_df.child.replace(label_to_taxonomi_dict))
    
            p_edges_df = p_edges_schima(p_edges_df)
            pre_detail_list = original_account_list_schima(locators_df)
            dict_t['status'] = 'success'            
            dict_t['org_taxonomi_cnt'] = len(pre_detail_list.query("schima_taxonomi.str.contains(@identifier)"))
            dict_t['org_taxonomi_list'] = pre_detail_list.query("schima_taxonomi.str.contains(@identifier)").schima_taxonomi.to_list()
    except Exception as e:
        #print(e)
        label_to_taxonomi_dict = {}
        dict_t['status'] = 'error'
        dict_t['error_message'] = e
        p_edges_df = p_edges_schima(pd.DataFrame(columns=get_columns(p_edges_schima)))
        pre_detail_list = original_account_list_schima(pd.DataFrame(columns=get_columns(original_account_list_schima)))
    
    return p_edges_df[get_columns(p_edges_schima)],pre_detail_list[get_columns(original_account_list_schima)],label_to_taxonomi_dict,dict_t

# %% main

filename = "../dataset/response_tbl_rst_2407_v1012.pkl"
response_tbl = pd.read_pickle(filename)#.set_index('docID')
# get original taxonomi list
dataset_summary_t = response_tbl#.head(1)#.tail(1000)
dict_all=[]
for docid in tqdm(dataset_summary_t.index):
    preproc_summary_series=response_tbl.loc[docid,:]

    out_path=PROJDIR / "data" / "2_intermediate" / ("data_pool_"+preproc_summary_series['dataset']) / docid
    p_edges_df,pre_detail_list,label_to_taxonomi_dict,dict_t=get_presentation_account_list_aud(
        docid=docid,
        identifier=preproc_summary_series.response_edinetCode,
        out_path=out_path)
    dict_all.append(dict_t)
org_list_df=pd.DataFrame(dict_all).query("org_taxonomi_cnt>0")
org_taxonomi_list_df=pd.Series(org_list_df.org_taxonomi_list.sum()).apply(lambda x: x.split('_')[-1]).value_counts()
org_taxonomi_list_df.to_csv(PROJDIR / "org_taxonomi_list_df_2407_v1012.csv")

""" Original taxonomi list
['FeeRelatedInformationConsolidatedTextBlock',
       'AuditorsResponseContinuedKAMConsolidatedTextBlock',
       'DescriptionIncludingReasonContinuedKAMConsolidatedTextBlock',
       'FeeRelatedInformationNonConsolidatedTextBlock',
       'DescriptionIncludingReasonContinuedKAMNonConsolidatedTextBlock',
       'AuditorsResponseContinuedKAMNonConsolidatedTextBlock',
       'AuditFirm3NonConsolidated',
       'DescriptionIncludingReasonContinued2KAMConsolidatedTextBlock',
       'AuditorsResponseContinued2KAMConsolidatedTextBlock',
       'CPA1AuditFirm3NonConsolidated', 'Reference6KAMConsolidated',
       'FeeRelatedInformationTextBlock',
       'IndependentAuditorsReportNonConsolidatedNATextBlock',
       'DescriptionIncludingReasonContinued2KAMNonConsolidatedTextBlock',
       'AuditorsResponseContinued2KAMNonConsolidatedTextBlock',
       'FeeReraltedInformationTextBlock', 'Reference7KAMConsolidated']
"""

add_aud_res_taxonomi_list = [
    'AuditorsResponseContinuedKAMConsolidatedTextBlock',
    'AuditorsResponseContinuedKAMNonConsolidatedTextBlock',
    'AuditorsResponseContinued2KAMConsolidatedTextBlock',
    'AuditorsResponseContinued2KAMNonConsolidatedTextBlock',
]

add_desc_taxonomi_list = [
    'DescriptionIncludingReasonContinuedKAMConsolidatedTextBlock',
    'DescriptionIncludingReasonContinuedKAMNonConsolidatedTextBlock',
    'DescriptionIncludingReasonContinued2KAMConsolidatedTextBlock',
    'DescriptionIncludingReasonContinued2KAMNonConsolidatedTextBlock',
    ]



# %%
dict_all=[]
log_dict_all=[]
dataset_summary_t=response_tbl
for docid in tqdm(dataset_summary_t.index):
    preproc_summary_series=response_tbl.loc[docid,:]
    out_path=PROJDIR / "data" / "2_intermediate" / ("data_pool_"+preproc_summary_series['dataset']) / docid
    periodEnd=preproc_summary_series['response_periodEnd']
    edinetCode=preproc_summary_series['response_edinetCode']
    xbrl_df,log_dict=get_xbrl_rapper_aud(docid,out_path,update_flg=False)
    log_dict_all.append(log_dict)
    if len(xbrl_df)>0:
        
        key_list=[
            'DescriptionIncludingReasonKAMConsolidatedTextBlock',
            'DescriptionIncludingReasonKAMNonConsolidatedTextBlock'
            ]+add_desc_taxonomi_list

        xbrl_df_f=xbrl_df.query("key.str.contains('|'.join(@key_list))")
        if len(xbrl_df_f)>0:
            xbrl_df_f=xbrl_df_f.assign(data_proc=xbrl_df_f.data_str.apply(data_utils.htmldrop).apply(preproc_nlp_kam))
            for itr_index in xbrl_df_f.index:
                dict_t={
                    'tag':'description',
                    'key':xbrl_df_f.loc[itr_index,'key'],
                    'text':xbrl_df_f.loc[itr_index,'data_proc'],
                    'docID':docid,
                    'context_ref':xbrl_df_f.loc[itr_index,'context_ref'],
                    'id':docid+'_'+xbrl_df_f.loc[itr_index,'context_ref'],
                    'periodEnd':periodEnd,
                    'edinetCode':edinetCode}
                dict_all.append(dict_t)        

        key_list_ref=[
            'AuditorsResponseKAMConsolidatedTextBlock',
            'AuditorsResponseKAMNonConsolidatedTextBlock',
            ]+add_aud_res_taxonomi_list
        xbrl_df_f=xbrl_df.query("key.str.contains('|'.join(@key_list_ref))")
        if len(xbrl_df_f)>0:
            xbrl_df_f=xbrl_df_f.assign(data_proc=xbrl_df_f.data_str.apply(data_utils.htmldrop).apply(preproc_nlp_kam_aud))
            for itr_index in xbrl_df_f.index:
                dict_t={
                    'tag':'audit_res',
                    'key':xbrl_df_f.loc[itr_index,'key'],
                    'text':xbrl_df_f.loc[itr_index,'data_proc'],
                    'docID':docid,
                    'context_ref':xbrl_df_f.loc[itr_index,'context_ref'],
                    'id':docid+'_'+xbrl_df_f.loc[itr_index,'context_ref'],
                    'periodEnd':periodEnd,
                    'edinetCode':edinetCode}
                dict_all.append(dict_t)

dict_all_df=pd.DataFrame(dict_all)
out_filename=PROJDIR / "dict_all_df_2407_v1012.csv"
dict_all_df.to_csv(out_filename,index=False)


log_dict_all_df=pd.DataFrame(log_dict_all)
out_filename=PROJDIR / "log_dict_all_df_2407_v1012.csv"
log_dict_all_df.to_csv(out_filename,index=False)

