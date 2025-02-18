
"""
env:dev?
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

#import importlib
#importlib.reload(preproc_rst_loader)


PROJPATH=r"/Users/noro/Documents/Projects/XBRL_common_space_projection/"
PROJDIR=Path(PROJPATH)
TESTPATH=r"/Users/noro/Documents/Projects/XBRL_common_space_projection/tests/20240313/"
# %%


# %% funcs

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
        aaa=(out_path / "audit_xbrl_proc_pd222.csv").exists()
        if (aaa)&(update_flg==False):
            xbrl_processed=pd.read_csv(out_path / "audit_xbrl_proc_pd222.csv",dtype=dtype(xbrl_elm_schima))
            return  xbrl_elm_schima(xbrl_processed),log_dict
    except Exception as e:
        aaa=False
        pass
    if (~aaa)|(update_flg==True):
#    if ((out_path / "audit_xbrl_proc_pd222.csv").exists())&(update_flg==False):
#        xbrl_processed=pd.read_csv(out_path / "audit_xbrl_proc_pd222.csv",dtype=dtype(xbrl_elm_schima))
#        return xbrl_elm_schima(xbrl_processed),log_dict
#    else:
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

filename=PROJDIR / "data/0_metadata/dataset_2407/response_tbl_rst_2407_v1012.pkl"
response_tbl=pd.read_pickle(filename)#.set_index('docID')
# %%

#
#dataset_summary_t = response_tbl#.head(1)#.tail(1000)
#dict_all=[]
#for docid in tqdm(dataset_summary_t.index):
#    preproc_summary_series=response_tbl.loc[docid,:]
#
#    out_path=PROJDIR / "data" / "2_intermediate" / ("data_pool_"+preproc_summary_series['dataset']) / docid
#    p_edges_df,pre_detail_list,label_to_taxonomi_dict,dict_t=get_presentation_account_list_aud(
#        docid=docid,
#        identifier=preproc_summary_series.response_edinetCode,
#        out_path=out_path)
#    dict_all.append(dict_t)
#org_list_df=pd.DataFrame(dict_all).query("org_taxonomi_cnt>0")
#org_taxonomi_list_df=pd.Series(org_list_df.org_taxonomi_list.sum()).apply(lambda x: x.split('_')[-1]).value_counts()
#org_taxonomi_list_df.to_csv(PROJDIR / "data/3_processed/dataset_2310/downstream/org_taxonomi_list_df_2407_v1012.csv")

# %% 1113 memo

# XBRLファイルにタグが入っていない...手動?

#dataset_summary_t = #.head(1)#.tail(1000)  'S100TUN9' 'S100OHIA'
#dict_all=[]
#for docid in tqdm(['S100LT42']):
#    preproc_summary_series=response_tbl.loc[docid,:]
#
#    out_path=PROJDIR / "data" / "2_intermediate" / ("data_pool_"+preproc_summary_series['dataset']) / docid
#    p_edges_df,pre_detail_list,label_to_taxonomi_dict,dict_t=get_presentation_account_list_aud(
#        docid=docid,
#        identifier=preproc_summary_series.response_edinetCode,
#        out_path=out_path)
#    dict_all.append(dict_t)
#org_list_df=pd.DataFrame(dict_all).query("org_taxonomi_cnt>0")
#org_taxonomi_list_df=pd.Series(org_list_df.org_taxonomi_list.sum()).apply(lambda x: x.split('_')[-1]).value_counts()
#{'S100LT42', 'S100OHIA', 'S100TUN9'}
#filename=PROJDIR / "data/3_processed/dataset_2310/downstream/1_raw/org_taxonomi_list_df_2407_v1012.csv"
#org_taxonomi_list_df=pd.read_csv(filename)



# %%
"""
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
    #'FeeRelatedInformationConsolidatedTextBlock',
    'AuditorsResponseContinuedKAMConsolidatedTextBlock',
    #'DescriptionIncludingReasonContinuedKAMConsolidatedTextBlock',
    #'FeeRelatedInformationNonConsolidatedTextBlock',
    #'DescriptionIncludingReasonContinuedKAMNonConsolidatedTextBlock',
    'AuditorsResponseContinuedKAMNonConsolidatedTextBlock',
    #'AuditFirm3NonConsolidated',
    'AuditorsResponseContinued2KAMConsolidatedTextBlock',
    #'DescriptionIncludingReasonContinued2KAMConsolidatedTextBlock',
    #'Reference6KAMConsolidated',
    #'CPA1AuditFirm3NonConsolidated',
    #'FeeRelatedInformationTextBlock',
    #'DescriptionIncludingReasonContinued2KAMNonConsolidatedTextBlock',
    'AuditorsResponseContinued2KAMNonConsolidatedTextBlock',
    #'FeeReraltedInformationTextBlock',
    #'Reference7KAMConsolidated'
]
#add_aud_res_taxonomi_list=list(map(str.lower,add_aud_res_taxonomi_list))

add_desc_taxonomi_list = [
    #'FeeRelatedInformationConsolidatedTextBlock',
    #'AuditorsResponseContinuedKAMConsolidatedTextBlock',
    'DescriptionIncludingReasonContinuedKAMConsolidatedTextBlock',
    #'FeeRelatedInformationNonConsolidatedTextBlock',
    'DescriptionIncludingReasonContinuedKAMNonConsolidatedTextBlock',
    #'AuditorsResponseContinuedKAMNonConsolidatedTextBlock',
    #'AuditFirm3NonConsolidated',
    #'AuditorsResponseContinued2KAMConsolidatedTextBlock',
    'DescriptionIncludingReasonContinued2KAMConsolidatedTextBlock',
    #'Reference6KAMConsolidated',
    #'CPA1AuditFirm3NonConsolidated',
    #'FeeRelatedInformationTextBlock',
    'DescriptionIncludingReasonContinued2KAMNonConsolidatedTextBlock',
    #'AuditorsResponseContinued2KAMNonConsolidatedTextBlock',
    #'FeeReraltedInformationTextBlock',
    #'Reference7KAMConsolidated'
    ]
#add_desc_taxonomi_list=list(map(str.lower,add_desc_taxonomi_list))



# %%
dict_all=[]
log_dict_all=[]
dataset_summary_t=response_tbl#.tail(200)
for docid in tqdm(dataset_summary_t.index):
    preproc_summary_series=response_tbl.loc[docid,:]
    out_path=PROJDIR / "data" / "2_intermediate" / ("data_pool_"+preproc_summary_series['dataset']) / docid
    periodEnd=preproc_summary_series['response_periodEnd']
    #periodEnd=preproc_summary_series['periodEnd']
    edinetCode=preproc_summary_series['response_edinetCode']
    #edinetCode=preproc_summary_series['edinetCode']
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

        #key_list_ref=[
        #    'jpcrp_cor:auditorsresponsekamconsolidatedtextblock',
        #    'jpcrp_cor:auditorsresponsekamnonconsolidatedtextblock',
        #    ]
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
out_filename=PROJDIR / "data/3_processed/dataset_2310/downstream/1_raw" / "dict_all_df_2407_v1012.csv"
dict_all_df.to_csv(out_filename,index=False)


log_dict_all_df=pd.DataFrame(log_dict_all)
out_filename=PROJDIR / "data/3_processed/dataset_2310/downstream/1_raw" / "log_dict_all_df_2407_v1012.csv"
log_dict_all_df.to_csv(out_filename,index=False)




# %% 2025/1/13 ###############################################################################################################
#
# data lineageの作成
#
##############################################################################################################################
from src.data.libs.utils import DataLinageJson
# %% 0831 input file
# markdown conversion was done for v831 version (run following 2 code once)
file_path = PROJDIR / "data/3_processed/dataset_2310/downstream/1_raw" / "dict_all_df_2407_v831.csv"
old_dict_all_df=pd.read_csv(file_path)
old_dict_all_df.head(1)


# %%
# convert ts to str

# %%
assertion_text = """
"""

processing_text = """
    1. presentation リンクベースファイル(pre.xml)から、オリジナルの要素名を取得
    2. オリジナルの要素名から、KAMとして取得する要素名を抽出
        # auditor response
        add_aud_res_taxonomi_list = [
            'AuditorsResponseContinuedKAMConsolidatedTextBlock',
            'AuditorsResponseContinuedKAMNonConsolidatedTextBlock',
            'AuditorsResponseContinued2KAMConsolidatedTextBlock',
            'AuditorsResponseContinued2KAMNonConsolidatedTextBlock',
            ]
        # description of KAM
        add_desc_taxonomi_list = [
            'DescriptionIncludingReasonContinuedKAMConsolidatedTextBlock',
            'DescriptionIncludingReasonContinuedKAMNonConsolidatedTextBlock',
            'DescriptionIncludingReasonContinued2KAMConsolidatedTextBlock',
            'DescriptionIncludingReasonContinued2KAMNonConsolidatedTextBlock',
            ]
    3. 以下のEDINETタクソノミに提出者オリジナルタクソノミを追加
        description: [
            'DescriptionIncludingReasonKAMConsolidatedTextBlock',
            'DescriptionIncludingReasonKAMNonConsolidatedTextBlock']
        audit_res: [
            'AuditorsResponseKAMConsolidatedTextBlock',
            'AuditorsResponseKAMNonConsolidatedTextBlock']
    4. textデータの前処理
        htmlタグの削除
        unicodedata.normalize("NFKC", text)
        replace("した","する")
"""
header_note_txt = """
    tag: description or audit_res
    key: prefix:element_name (KAMのEDINETタクソノミ)
    text: KAMのelement_nameに対応するテキストデータ
    docID:
    context_ref:
    id:
    periodEnd:
    edinetCode:
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
        "dict_all_df":[str(PROJDIR / "data/0_metadata/dataset_2407/response_tbl_rst_2407.pkl")],
        },
    "input_data_providing_func": {
        "dict_all_df":"",
        },
    "index_name": old_dict_all_df.index.name,
    "header": list(old_dict_all_df.columns),
    "count": len(old_dict_all_df),
    "unique_count_index": old_dict_all_df.index.nunique(),
    "unique_count_header": old_dict_all_df.describe(include='all').T['unique'].to_dict(),
    "example_rcd": old_dict_all_df.iloc[0].to_dict(),
    "header_note": header_note_txt,
    "src": "data/ds01_01_extract_kam_text.py",
    "assertion": "",
    "processing": processing_text,
    "note": ""
})
DataLinageJson_obj.save()

# %% 1012
file_path = PROJDIR / "data/3_processed/dataset_2310/downstream/1_raw" / "dict_all_df_2407_v1012.csv"
dict_all_df=pd.read_csv(file_path)
dict_all_df.head(1)
# %%

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
        "dict_all_df":[str(PROJDIR / "data/0_metadata/dataset_2407/response_tbl_rst_2407_v1012.pkl")],
        },
    "input_data_providing_func": {
        "dict_all_df":"",
        },
    "index_name": dict_all_df.index.name,
    "header": list(dict_all_df.columns),
    "count": len(dict_all_df),
    "unique_count_index": dict_all_df.index.nunique(),
    "unique_count_header": dict_all_df.describe(include='all').T['unique'].to_dict(),
    "example_rcd": dict_all_df.iloc[0].to_dict(),
    "header_note": header_note_txt,
    "src": "data/ds01_01_extract_kam_text.py",
    "assertion": "",
    "processing": processing_text,
    "note": ""
})
DataLinageJson_obj.save()






# %%
# %% TEST Arelle
#TESTDIR=Path(TESTPATH) 
#xbrl_filename="/Users/noro/Documents/Projects/XBRL_common_space_projection/tests/20240809/XBRL/AuditDoc/jpaud-aar-cn-001_E02542-000_2021-03-31_01_2021-06-25.xbrl"
#ctrl = Cntlr.Cntlr(logFileName=str(TESTDIR / "arelle.log"))
#model_xbrl = ctrl.modelManager.load(xbrl_filename)
## %%
#from arelle.ModelValue import qname
#model_xbrl.prefixedNamespaces
## %%
#localname="AccountingStandardsDEI"
#qname_prefix = "xbrli"
#ns = model_xbrl.prefixedNamespaces[qname_prefix]
#tmp=model_xbrl.factsByQname[qname(ns, name=f"{qname_prefix}:{localname}")]
#
#model_xbrl.factsByQname
#ctrl.close()
# %% memo


#filename=PROJDIR / "data/3_processed/dataset_2310/downstream" / "dict_all_df_2407_v821.csv"
#dict_all_df=pd.read_csv(filename)
#
#filename=PROJDIR / "data/3_processed/dataset_2310/downstream" / "log_dict_all_df_2407_v821.csv"
#log_dict_all_df=pd.read_csv(filename)

# %%
#log_dict_all_df.status.value_counts()

# %% memo
#log_dict_all_df.query("status=='Failure'").error_message.value_counts()
# %%

#filename=PROJDIR / "data/3_processed/dataset_2310/downstream" / "0816"/"dict_all_df_2407_v816.csv"
#dict_all_df_old=pd.read_csv(filename)
#
#new_docID=set(dict_all_df.docID)
# %%
#
#filename=PROJDIR / "data/3_processed/dataset_2310/downstream" / "0816"/"dict_all_df_2407_v816.csv"
#dict_all_df_old=pd.read_csv(filename)
#
## %%
#old_docID=set(dict_all_df_old.docID)
### %%
#diff_docID=new_docID-old_docID
#dict_all_df.query("docID in @diff_docID").shape


# %%
#dataframes = []
#dict_all=[]
#        
#dataset_summary_t=dataset_summary.sample(1,random_state=0)#query("docID=='S100R6CC'")
#for docid in tqdm(dataset_summary_t.index):
#    preproc_summary_series=dataset_summary.loc[docid,:]
#
#    out_path=PROJDIR / "data" / "2_intermediate" / ("data_pool_"+preproc_summary_series['dataset']) / docid
#
#    data_dir_raw=PROJDIR / "data" / "1_raw"
#    zip_file = list(data_dir_raw.glob("data_pool_*/"+docid+".zip"))[0]
#    #zip_file = list(self.data_dir_law.glob("data_pool_*/"+self.docid+".zip"))[0]
#    with ZipFile(str(zip_file)) as zf:
#        #print(zf.namelist())
#        fn=[item for item in zf.namelist() if (".xbrl" in item)&("aai" in item)]
#        if len(fn)>0:
#            zf.extract(fn[0], out_path)
#    xbrl_path=out_path / "XBRL" / "AuditDoc"
#    if len(list(xbrl_path.glob("*.xbrl")))>0: # xbrl file exists
#        xbrl_filename=str(list(xbrl_path.glob("*.xbrl"))[0])
#        xbrl_df=get_xbrl_df(xbrl_filename)
#        key_list=[
#            'jpcrp_cor:descriptionincludingreasonkamconsolidatedtextblock',
#            'jpcrp_cor:descriptionincludingreasonkamnonconsolidatedtextblock'
#            ]
#        xbrl_df_f=xbrl_df.query("key in @key_list")
#        if len(xbrl_df_f)>0:
#            for itr_text in xbrl_df_f.data.apply(data_utils.htmldrop).apply(preproc_nlp_kam).to_list():
#                dict_t={'tag':'long','text':itr_text,'docID':docid,
#                        }
#                dict_all.append(dict_t)
#        key_list_short=[
#            'jpcrp_cor:shortdescriptionkamconsolidated',
#            'jpcrp_cor:shortdescriptionkamnonconsolidated',
#            ]
#        xbrl_df_f=xbrl_df.query("key in @key_list_short")
#        if len(xbrl_df_f)>0:
#            for itr_text in xbrl_df_f.data.apply(data_utils.htmldrop).apply(preproc_nlp_kam).to_list():
#                dict_t={'tag':'short','text':itr_text,'docID':docid}
#                dict_all.append(dict_t)
#        key_list_ref=[
#            'jpcrp_cor:referencekamconsolidated',
#            'jpcrp_cor:referencekamnonconsolidated',
#            ]
#        xbrl_df_f=xbrl_df.query("key in @key_list_ref")
#        if len(xbrl_df_f)>0:
#            for itr_text in xbrl_df_f.data.apply(data_utils.htmldrop).apply(preproc_nlp_kam).to_list():
#                dict_t={'tag':'ref','text':itr_text,'docID':docid}
#                dict_all.append(dict_t)
#


# %%
#from edinet_xbrl.edinet_xbrl_parser import EdinetXbrlParser
#parser = EdinetXbrlParser()
#
#def get_xbrl_df2(xbrl_filename:str)->pd.DataFrame:
#    edinet_xbrl_object = parser.parse_file(xbrl_filename)
#    rst=[]
#    for itr_key in edinet_xbrl_object.get_keys():
#        context_ref_list=edinet_xbrl_object.get_data_list(itr_key)
#        for itr_context_ref in context_ref_list:
#            value_extracted=edinet_xbrl_object.get_data_by_context_ref(itr_key,itr_context_ref.get_context_ref()).get_value()
#            rst.append({"key":itr_key,"context_ref":itr_context_ref.get_context_ref(),"data":value_extracted})
#
#    rst_df=pd.DataFrame(rst).drop_duplicates()
#    
#    return rst_df

# %%

#for docid in tqdm(dataset_summary_t.index):
#
#    proc_rst = preproc_summary_series.to_dict()
#    try:
#        #docid='S100LGTD'
#        out_path = PROJDIR / "data" / "2_intermediate" / ("data_pool_"+preproc_summary_series['dataset']) / docid
#        data_dir_raw = PROJDIR / "data" / "1_raw"
#        
#        zip_file = list(data_dir_raw.glob("data_pool_*/"+docid+".zip"))[0]
#        with ZipFile(str(zip_file)) as zf:
#                fn=[item for item in zf.namelist() if ("pre.xml" in item)&("aai" in item)]
#                if len(fn)>0:
#                    zf.extract(fn[0], out_path)
#        xml_def_path=out_path / "XBRL" / "AuditDoc"
#        tree = ET.parse(str(list(xml_def_path.glob("*pre.xml"))[0]))
#        root = tree.getroot()
#
#        locators = []
#        arcs = []
#        for child in root:
#            attr_sr_p = pd.Series(child.attrib)
#            role = attr_sr_p[attr_sr_p.index.str.contains('role')].values[0]
#            for child_of_child in child:
#                locator = {'role':role,'schima_taxonomi':None}
#                arc = {'parent':None,'child':None,'child_order':None,'fs':role}
#
#                attr_sr = pd.Series(child_of_child.attrib)
#                attr_type = attr_sr[attr_sr.index.str.contains('type')].values[0]
#                if attr_type=='locator':
#                    locator['schima_taxonomi'] = attr_sr[attr_sr.index.str.contains('href')].values[0].split('#')[1]
#                    locator['label'] = attr_sr[attr_sr.index.str.contains('label')].values[0]
#                elif attr_type=='arc':
#                    arc['parent'] = attr_sr[attr_sr.index.str.contains('from')].values[0]
#                    arc['child'] = attr_sr[attr_sr.index.str.contains('to')].values[0]
#                    arc['child_order'] = attr_sr[attr_sr.index.str.contains('order')].values[0]
#
#                locators.append(locator)
#                arcs.append(arc)
#
#        locators_df = pd.DataFrame(locators).dropna(subset=['schima_taxonomi'])
#        locators_df = locators_df.assign(rol=locators_df.role.str.split('/',expand=True).iloc[:,-1],
#                                        taxonomi_tag=locators_df.schima_taxonomi.str.lower())
#        label_to_taxonomi_dict = locators_df.set_index('label')['taxonomi_tag'].to_dict()
#        camel_taxonomi_dict = locators_df.set_index('taxonomi_tag')['schima_taxonomi'].to_dict()
#
#        p_edges_df2 = pd.DataFrame(arcs).dropna(subset=['child'])
#        p_edges_df = pd.DataFrame(arcs).dropna(subset=['child'])
#        p_edges_df = p_edges_df.assign(parent_taxonomi_tag=p_edges_df.parent.replace(label_to_taxonomi_dict),
#            child_taxonomi_tag = p_edges_df.child.replace(label_to_taxonomi_dict))
#
#        p_edges_df = p_edges_schima(p_edges_df)
#        label_to_taxonomi_dict = label_to_taxonomi_dict
#        camel_taxonomi_dict = camel_taxonomi_dict
#        pre_detail_list = original_account_list_schima(locators_df)
#        proc_rst['load_pre'] = 'success'
#        dict_t={
#            'docID':docid,
#            'org_taxonomi_cnt':len(pre_detail_list.query("schima_taxonomi.str.contains(@preproc_summary_series.parse_identifier)")),
#            'org_taxonomi_list':pre_detail_list.query("schima_taxonomi.str.contains(@preproc_summary_series.parse_identifier)").schima_taxonomi.to_list()
#                }
#        dict_all.append(dict_t)
#    except Exception as e:
#        #print(e)
#        proc_rst['load_pre'] = e
#        label_to_taxonomi_dict = {}
#        p_edges_df = p_edges_schima(pd.DataFrame(columns=['parent_taxonomi_tag','child_taxonomi_tag','fs']))
#        pre_detail_list = original_account_list_schima(pd.DataFrame(columns=['schima_taxonomi','label','taxonomi_tag','rol']))
## %%