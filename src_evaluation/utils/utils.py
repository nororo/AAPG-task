

########################################################################
#
# utils for pandera schima
#
########################################################################


import pandera as pa
from pandera.typing import DataFrame, Series
import pandas as pd

def get_columns(schima:pa.DataFrameModel)->list:
    return list(schima.to_schema().columns.keys())

def dtype(df_type, use_nullable=True):
    dc = {}
    schema = df_type.to_schema()
    for name, column in schema.columns.items():
        typ = column.dtype.type
        if use_nullable and column.nullable and column.dtype.type == int:
            typ = "Int64"
        dc[name] = typ
    return dc

########################################################################
#
# list processing
#
########################################################################


def remove_empty_lists(lst):
    return [x for x in lst if x]

def flatten_list(lst):
    flat_list = []
    for item in lst:
        if isinstance(item, list):
            flat_list.extend(flatten_list(item))
        else:
            flat_list.append(item)
    return flat_list

########################################################################
#
# multi processing
#
########################################################################


import contextlib
from tqdm import tqdm

@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """
    Context manager to patch joblib to report into tqdm progress bar given as argument
    https://stackoverflow.com/questions/24983493/tracking-progress-of-joblib-parallel-execution/58936697#58936697
    """
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()

########################################################################
#
# Timer
#
########################################################################



import time
@contextlib.contextmanager
def timer(name):
    t0=time.time()
    yield
    print(f'[{name}] done in {time.time()-t0:.2f} s ')

def read_data_linage(filename:str)->dict:
    assert Path(filename).is_file(), f"File not found: {filename}"

    file_name_wo_ext = Path(filename).stem
    dir_name = Path(filename).parent
    file_name = f"{file_name_wo_ext}.json"
    file_path = dir_name / file_name
    with open(file_path, "r") as f:
        data = json.load(f)
    return data


########################################################################
#
# Text preprocessing
#
########################################################################

import unicodedata
import string
import urllib.request as request
import re

def htmldrop(text:str)->str:
    return re.sub(re.compile('<.*?>'), '', text)


def RtnDroper(text:str)->str:
    replaced_text=text.replace('\n\n','\n')
    replaced_text=replaced_text.replace('\n \n','\n')
    if replaced_text==text:
        return replaced_text
    else:
        return RtnDroper(replaced_text)


def get_stopwards()->list:
    sw_filename = PROJDIR / "data/0_metadata/text_emb/prep/stopwords.txt"
    if not Path(sw_filename).exists():
        res = request.urlopen("http://svn.sourceforge.jp/svnroot/slothlib/CSharp/Version1/SlothLib/NLP/Filter/StopWord/word/Japanese.txt").read().decode("utf-8")
        with open(sw_filename, "w") as f:
            f.write(res)
    else:
        with open(sw_filename) as f:
            res = f.read()
    stopwords = [line.strip() for line in res.split("\n")]
    return stopwords



def preproc_text(text: str) -> str:
    replaced_text = unicodedata.normalize("NFKC", text)
    #replaced_text=replaced_text.replace('\n','')
    replaced_text=replaced_text.replace(' ','')
    replaced_text=RtnDroper(replaced_text)
    replaced_text=SpaceDroper(replaced_text)
    replaced_text = re.sub(re.compile("[!-/:-@[-`{-~]"), '', replaced_text)
    # drop signature 2
    replaced_text = re.sub(r'\(', '', replaced_text)
    # drop signature 3
    table = str.maketrans("", "", string.punctuation  + "◆■※【】)(「」、。・")
    replaced_text = replaced_text.translate(table)
    return replaced_text

def drop_number(text:str)->str:
    """
    pattern = r'\d+'
    replacer = re.compile(pattern)
    result = replacer.sub('0', text)
    """
    # 連続した数字を0で置換
    replaced_text = re.sub(r'\d+', '', text)
    
    return replaced_text

def SpaceDroper(text):
    replaced_text=text.replace('  ',' ')
    if replaced_text==text:
        return replaced_text
    else:
        return RtnDroper(replaced_text)

# %% llm output

import json
def extract_json_dict_old(text:str,required_output_num=2)->list:
    if len(text)==0:
        ValueError("text is empty")
    text=re.sub(r'\n', '',text)
    pattern = '{.*?}'
    text_json_list=re.findall(pattern, text)
    item_list=[]
    for text_json in text_json_list:
        tmp_dict=json.loads(text_json)
        item_list.append(tmp_dict)
    vals=list(item_list[0].values())
    if len(vals)!=required_output_num:
        ValueError("output number is not correct")
    if len(vals)==1:
        return vals[0]
        
    if len(vals)==2:
        pred=vals[0]
        reason=vals[1]
        return pred,reason

    elif len(vals)==3:
        pred=vals[0]
        confidence_score=vals[1]
        reason=vals[2]
        return pred,confidence_score,reason
    else:
        return vals

def extract_json_dict(text:str)->list:
    if len(text)==0:
        ValueError("text is empty")
    text=re.sub(r'\n', '',text)
    pattern = '{.*?}'
    text_json_list=re.findall(pattern, text)
    item_list=[]
    for text_json in text_json_list:
        tmp_dict=json.loads(text_json)
        item_list.append(tmp_dict)

    return item_list

from libs.compose_prompt import *
from libs.utils import *


EVAL_MODEL="gpt_4o_mini"
class batch_inf_file_generator():
    def __init__(self,prompt_dict=dict(),make_prompt_func="",model_name="gpt_4o_mini",max_tokens=1024):
        self.prompt_dict=prompt_dict
        self.inf_list=[]
        self.make_prompt=make_prompt_func
        self.model_name=model_name
        self.model_dict={
            "gpt_4o":"gpt-4o-2024-08-06", # 2023/10 I$2.5/O$10
            "gpt_4o_mini":"gpt-4o-mini-2024-07-18", # 2023/8 I$0.15/O$0.6
            "gpt_4_turbo":"gpt-4-turbo-2024-04-09", # 2023/12 I$10/O$30
            "gpt_4":"gpt-4", # 2023/12 I$30/O$60
            "gpt_3.5":"gpt-3.5-turbo-0125", # 2021/9 I$0.5/O$1.5
            "llama_3.1_70b": "llama-3.1-70b-versatile",
            "llama_3.1_8b": "llama-3.1-8b-instant",
            "llama_3_70b": "llama3-70b-8192",
            "llama_3_8b": "llama3-8b-8192",
            "gemma_7b": "gemma-7b-it",
            "gemma_2_9b": "gemma2-9b-it",
            "llava":"llava-v1.5-7b-4096-preview"            
            }
        self.temperature_dict={
            "gpt_4o":1,
            "gpt_4o_mini":1,
            "gpt_4_turbo":1,
            "gpt_4":1,
            "gpt_3.5":1,
            "llama_3.1_70b": 0.6,
            "llama_3.1_8b": 0.6,
            "llama_3_70b": 0.6,
            "llama_3_8b": 0.6,
            "gemma_7b": 0.6,
            "gemma_2_9b": 0.6,
            "llava": 0.6
            }

    def insert_inf_list(self,text,itr_index_num,max_tokens=1024):
        sys_prompt, usr_prompt = self.make_prompt(self.prompt_dict,text)
    
        messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": usr_prompt},
            ]
        temp={
            "custom_id": "request-"+str(itr_index_num),
            "method": "POST", "url": "/v1/chat/completions",
            "body": {
                "model": self.model_dict[self.model_name],
                "messages":messages,
                "temperature": 1,
                "max_tokens": max_tokens
                }}
        self.inf_list.append(temp)

    def insert_inf_list_prompt(self,sys_prompt,usr_prompt,itr_index_num,max_tokens=1024,model_name=""):
        if len(model_name)==0:
            model_name=self.model_name
        temperature=self.temperature_dict[model_name]
        messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": usr_prompt},
            ]
        temp={
            "custom_id": "request-"+str(itr_index_num),
            "method": "POST", "url": "/v1/chat/completions",
            "body": {
                "model": self.model_dict[model_name],
                "messages":messages,
                "temperature": temperature,
                "max_tokens": max_tokens
                }}
        self.inf_list.append(temp)

    def export_list(self,out_filename):

        with open(out_filename, 'w') as file:
            for obj in self.inf_list:
                file.write(json.dumps(obj) + '\n')
    
    def print_sample(self):
        print("--- System Prompt: ---")
        print(self.inf_list[-1]["body"]["messages"][0]["content"])
        print("--- User Prompt: ---")
        print(self.inf_list[-1]["body"]["messages"][1]["content"])

# same?
def get_results(filename_openai_rst):
    batch_output = pd.read_json(filename_openai_rst, orient='records', lines=True)
    ans_list=[]
    for itr_index in range(len(batch_output.response)):
        ans_t={
                'index_num':str(itr_index),
                'output':'-',
                'api_status':'-',
                'pred':'-',
                'status':'-',
                }
        output=batch_output.response[itr_index]['body']['choices'][0]['message']['content']
        #ans_t["output"]=output#["output"]
        #ans_t["api_status"]=output["status"]
        try:
            out_json_list=extract_json_dict(output)
            ans_t["output"]=out_json_list
            ans_t['status']='Success'
        except Exception as e:
            print(e)
            ans_t['status']='Failed'
        ans_list.append(ans_t)
    return ans_list

def get_results_openai_batch(filename_openai_rst,json=True):
    batch_output = pd.read_json(filename_openai_rst, orient='records', lines=True)
    ans_list=[]
    for itr_index in range(len(batch_output.response)):
        ans_t={
                'index_num':batch_output.custom_id[itr_index],
                'output':'-',
                'api_status':'-',
                'pred':'-',
                'status':'-',
                }
        
        output=batch_output.response[itr_index]['body']['choices'][0]['message']['content']
        #ans_t["output"]=output#["output"]
        #ans_t["api_status"]=output["status"]
        try:
            if json:
                out_json_list=extract_json_dict(output)
                ans_t["output"]=out_json_list
            else:
                ans_t["output"]=output
            ans_t['status']='Success'
        except Exception as e:
            print(e)
            ans_t['status']='Failed'
        ans_list.append(ans_t)
    return ans_list

from typing import Annotated
from pydantic import BaseModel, Field,SecretStr
from pydantic.functional_validators import BeforeValidator
from pathlib import Path

StrOrNone = Annotated[str, BeforeValidator(lambda x: x or "")]
IntOrNone = Annotated[int, BeforeValidator(lambda x: x or 0)]

class DataLinageJson(BaseModel):
    """
    """
    create_date: str = Field(description="作成日")
    check_date: str = Field(description="チェック日")
    size: str = Field(description="ファイルサイズ")
    file_path: str = Field(description="ファイルパス")
    input_data: dict
    input_data_providing_func: dict
    index_name: StrOrNone = Field(description="index")
    header: list = Field(description="header")
    count: int = Field(description="count")
    unique_count_index: IntOrNone = Field(description="unique count index")
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
        file_name = f"{file_name_wo_ext}_lin.json"
        print(dir_name / file_name)
        with open(dir_name / file_name, 'w') as f:
            f.write(self.json())
