# %%
import csv
from transformers import AutoTokenizer, AutoModel
import torch
from torch import Tensor
import torch.nn.functional as F
from tqdm import tqdm


# %%

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
PROJPATH=r"/Users/noro/Documents/Projects/XBRL_common_space_projection/"
PROJDIR=Path(PROJPATH)

filename=PROJDIR / "data/3_processed/dataset_2310/downstream" / "all_data_mapping" /"dict_all_df_2407_v1012_vect_df.pkl"
vectors_df=pd.read_pickle(filename)
# %%
import umap
from sklearn.cluster import HDBSCAN
reducer = umap.UMAP(n_components=2, n_neighbors=50, random_state=0)
reducer.fit(vectors_df)
# %%
comp_embedding = reducer.transform(vectors_df)

# %%
np.random.seed(0)
hdb = HDBSCAN(min_cluster_size=100,max_cluster_size=None,store_centers="both")
hdb.fit(comp_embedding)
cls_labels=pd.Series(hdb.labels_,index=vectors_df.index)
cls_labels.value_counts()

# %% 20250119 get importace from hdb
#dir(hdb)
filename=PROJDIR / "data/3_processed/dataset_2310/downstream/all_data_mapping" / "data_all17k_pivot_2407_v1012_with_cls.csv"
data_all_pivot=pd.read_csv(filename)

comp_embedding_df = pd.DataFrame(comp_embedding, index=data_all_pivot.id)
comp_embedding_df.head()
# %%

filename=PROJDIR / "data/3_processed/dataset_2310/downstream" / "3_processed/sft_data" /"data_train_1012.csv"
data_train = pd.read_csv(filename).head()
# %%
data_train
#hdb.centroids_
vectors_df
data_all_pivot

# %%
hdb.medoids_




# %%

#filename=PROJDIR / "data/3_processed/dataset_2310/downstream" / "data_all_pivot_2407_v1012.csv"
filename=PROJDIR / "data/3_processed/dataset_2310/downstream/all_data_mapping" / "data_all17k_pivot_2407_v1012.csv"
data_all_pivot=pd.read_csv(filename)
#data_all_pivot=data_all_pivot.assign(cls_labels=cls_labels)
# %%
out_filename=PROJDIR / "data/3_processed/dataset_2310/downstream/all_data_mapping" / "data_all17k_pivot_2407_v1012_with_cls.csv"
data_all_pivot.to_csv(out_filename,index=False)
#comp_embedding.head()
# %%
filename=PROJDIR / "data/3_processed/dataset_2310/downstream" / "all_data_mapping" /"dict_all_df_2407_v1012_vect_df_2dim.pkl"
pd.DataFrame(comp_embedding).to_pickle(filename)

# %%
import matplotlib.pyplot as plt
import seaborn as sns
sns.scatterplot(pd.DataFrame(comp_embedding).assign(cls_labels=cls_labels2.values),x=0,y=1,hue='cls_labels')#, hue=["cluster-{}".format(x) for x in cls_labels])
plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
#plt.scatter(comp_embedding[:,0],comp_embedding[:,1],c=
#c=cls_labels,s=1,alpha=0.5)

# %%


topic_name_dict={
    -1:"その他",
    0:"税効果会計、グループ納税",
    1:"有価証券や金融資産の評価",
    2:"ソフトウェアの評価",
    3:"棚卸資産の評価",#
    4:"棚卸資産（販売用不動産）の評価",
    5:"固定資産の減損",
    6:"のれん、企業結合",
    7:"不正、内部統制の不備への対応",
    8:"継続企業の前提の検討",#
    9:"不動産の売却",#
    10:"貸倒引当金の評価",
    11:"収益認識",
    12:"偶発債務、偶発損失、引当金（保証損失）",#
    13:"引当金（受注損失、工事損失）、見積原価の不確実性"
}

#topic_name_dict2={
#    -1:"その他",
#    0:"税効果会計、グループ納税",
#    1:"不動産の売却",
#    2:"有価証券や金融資産の評価",
#    3:"ソフトウェアの評価",#
#    4:"棚卸資産（販売用不動産）の評価",
#    5:"棚卸資産の評価",
#    6:"継続企業の前提の検討",
#    7:"不正、内部統制の不備への対応",
#    8:"のれんの評価",#
#    9:"固定資産の減損",#
#    10:"貸倒引当金の評価",
#    11:"収益認識",
#    12:"偶発債務、偶発損失、引当金（保証損失）",#
#    13:"引当金（受注損失、工事損失）、見積原価の不確実性"
#}

cls_labels2=cls_labels.map(topic_name_dict)
cls_sort=cls_labels2.value_counts()
cls_sort
# %%
# %%
itr=13
data_all_pivot.query("cls_labels==@itr").head(50).description
n=45

print(data_all_pivot.query("cls_labels==@itr").head(50).description.iloc[n+1])
print('\n\n 2 \n\n')
print(data_all_pivot.query("cls_labels==@itr").head(50).description.iloc[n+2])
print('\n\n 3 \n\n')
print(data_all_pivot.query("cls_labels==@itr").head(50).description.iloc[n+3])

# %% postprocess




filename=PROJDIR / "data/3_processed/dataset_2310/downstream" / "all_data_mapping" /"dict_all_df_2407_v1012_vect_df_2dim.pkl"
comp_embedding=pd.read_pickle(filename)
# %%
filename=PROJDIR / "data/3_processed/dataset_2310/downstream/all_data_mapping" / "data_all17k_pivot_2407_v1012_with_cls.csv"
cls_labels_df=pd.read_csv(filename)#.cls_labels
# %%
#comp_embedding
#cls_labels

topic_name_dict={
    -1:"その他",
    0:"税効果会計、グループ納税",
    1:"有価証券や金融資産の評価",
    2:"ソフトウェアの評価",
    3:"棚卸資産の評価",#
    4:"棚卸資産（販売用不動産）の評価",
    5:"固定資産の減損",
    6:"のれん、企業結合",
    7:"不正、内部統制の不備への対応",
    8:"継続企業の前提の検討",#
    9:"不動産の売却",#
    10:"貸倒引当金の評価",
    11:"収益認識",
    12:"偶発債務、偶発損失、引当金（保証損失）",#
    13:"引当金（受注損失、工事損失）、見積原価の不確実性"
}
cls_labels2=cls_labels_df.cls_labels.map(topic_name_dict)
cls_labels2.value_counts()


#pd.DataFrame(comp_embedding).to_pickle(filename)







# %%
from libs.sudachi_tokenizer import *
from src.data import data_utils

# %%
sudachipy_tokenizer = SudachiTokenizer(stopwords=data_utils.get_stopwards())
# %%
cls_labels_df=cls_labels_df.assign(description_sep=cls_labels_df.description.apply(preproc_add_forlda))

cls_labels_df=cls_labels_df.assign(description_tokenized=cls_labels_df['description_sep'].progress_apply(tokenize_sudachi_for_series))

# %%
cls_labels_df=cls_labels_df.assign(audit_res_sep=cls_labels_df.audit_res.apply(preproc_add_forlda))

cls_labels_df=cls_labels_df.assign(audit_res_tokenized=cls_labels_df['audit_res_sep'].progress_apply(tokenize_sudachi_for_series))

# %% SFT check
itr=13
filename="/Users/noro/Documents/Projects/XBRL_common_space_projection/results/downstream/kam/topic_sft1127/traing_log_history_1115_"+str(itr)+".json"
train_log=pd.read_json(filename)
train_log.query("eval_loss.notna()").set_index('epoch').eval_loss.plot()
min_idx=train_log.query("eval_loss.notna()").eval_loss.idxmin()
min_epoch=train_log.query("eval_loss.notna()").loc[min_idx,'epoch']
print(min_epoch)
#eval_loss.plot()

# %%
cls_labels_df=cls_labels_df.assign(len_text_tokenized=cls_labels_df['description_tokenized'].apply(len))

# %%
# %%

from gensim.corpora.dictionary import Dictionary


dictionary = Dictionary(data_all_pivot.description_tokenized.to_list())
dictionary[0]
dictionary.id2token
# %%
dictionary.filter_extremes(no_below=10, no_above=0.2,keep_n=400000)
dictionary.compactify() # Reassign id

corpus = [dictionary.doc2bow(itrtext) for itrtext in data_all_pivot.description_tokenized.to_list()]
dictionary[0]
dictionary.id2token

# %%
data_all_pivot=data_all_pivot.assign(corpus=corpus)
# %%
# %%


from googletrans import Translator

def translate_to_eng(word_list:list)->list:
    """
    Translate the word list to English by Google Translate.
    """
    jpnw='\n'.join(word_list)
    translator = Translator()
    engw=translator.translate(jpnw).text
    engwlist=engw.split('\n')
    return engwlist

word_list_eng=translate_to_eng(word_list.index.to_list())
# %%
#expolist.index=
#x=expolist.to_dict()



# %%
import wordcloud as wc 
from matplotlib import pyplot as plt

def plot_wordcloud(word_dict:dict,out_filename_pdf="",tran_eng=False)->None:
    fig=plt.figure(figsize=(10,10))
    font=r"/Users/noro/Documents/analysis/2019.3/書き出されたフォント/Hiragino Maru Gothic ProN/ヒラギノ丸ゴ ProN W4.ttc"
    
    if tran_eng:
        font=r"/Users/noro/Documents/Projects/XBRLanalysis/Arial/Arial/Arial.ttf"
        word_dict_sr=pd.Series(word_dict)
        word_dict_sr.index=translate_to_eng(word_dict_sr.index.to_list())
        word_dict=word_dict_sr.to_dict()

    wc_pc0=wc.WordCloud(
            background_color="white",
            font_path=font,
            max_words=50,
            min_font_size=10,max_font_size=96,width=800,height=600
            ).generate_from_frequencies(word_dict)
    ax=fig.add_subplot(1,1,1)
    ax.imshow(wc_pc0)
    ax.axis("off")
    if len(out_filename_pdf)>0:
        fig.savefig(out_filename_pdf, transparent=True)

for itr in range(14):
    new_dct = Dictionary.from_corpus(data_all_pivot.query("cls_labels==@itr").corpus.to_list(),id2word= dictionary.id2token)
    word_list=pd.DataFrame([(dictionary.id2token[key],new_dct.dfs[key]) for key in new_dct.dfs.keys()],columns=['word','dfs']).sort_values('dfs',ascending=False).head(50)
    word_list=word_list.set_index('word')
    word_list_dict=word_list.to_dict()['dfs']
    out_filename_pdf="/Users/noro/Documents/Projects/XBRL_common_space_projection/results/downstream/kam/kam_topic/wordcloud_topic_{}.pdf".format(itr)
    plot_wordcloud(word_list_dict,out_filename_pdf=out_filename_pdf)

# %%
import openai
# %%
from pathlib import Path
out_path=Path("/Users/noro/Documents/Projects/XBRL_common_space_projection/data/2_intermediate/data_pool_2003_120/S100A0A3/textblock_curs.csv")
#out_path=Path("/Users/noro/Documents/Projects/XBRL_common_space_projection/data/2_intermediate/data_pool_2407_120/S100SBR1/textblock_cur.csv")
already_exist_flg=out_path.exists()
already_exist_flg
# %%
from datetime import datetime
datetime.fromtimestamp(out_path.stat().st_mtime)#>datetime(2024,10,1)


# %%
from nltk import FreqDist
new_dct.dfs
fdist1 = FreqDist(text1)
print fdist1.most_common(50)

# %% gomi
import unicodedata
import string

def preproc_add_forlda(text:str)->str:
    # unicode
    replaced_text=text.replace('。','\n')
    # drop signature 3
    table = str.maketrans("", "", string.punctuation  + "◆■※【】)(「」、・")
    replaced_text = replaced_text.translate(table)
    
    return replaced_text


# %%

import sudachipy
import sudachidict_core


class SudachiTokenizer():
    def __init__(self, mode='C', stopwords=None):
        if mode == 'A':
            self.mode=sudachipy.SplitMode.A
        elif mode == 'B':
            self.mode=sudachipy.SplitMode.B
        elif mode == 'C':
            self.mode=sudachipy.SplitMode.C
        else:
            raise ValueError(f"Available input is A,B,C: {mode}.")
        
        self._tok = sudachipy.Dictionary(dict_type="core").create(mode=self.mode)
        self.include_part_of_speech_list=['名詞','形容詞']
        if stopwords is None:
            self.stopwords = []
        else:
            self.stopwords = stopwords
        return

    def tokenize(self, sent):
        
        ms = self._tok.tokenize(sent, self.mode)
        for m in ms:
            p = m.part_of_speech()
            m.normalized_form()
        return [m.normalized_form() for m in ms if m.part_of_speech()[0] in self.include_part_of_speech_list and m.normalized_form() not in self.stopwords]

from tqdm import tqdm
tqdm.pandas()
def tokenize_sudachi_for_series(article):
    """
    Tokenize the input text using SudachiPy.
    When input is too long, it can't be more than 49149 bytes.So, we split the input text by line and tokenize each line.
    https://zenn.dev/hyga2c/articles/ginza5_largetext
    """
    if len(article)==0:
        return []
    if len(article)>5000:
        tokenized_line=[]
        for line in article.split('\n'):
            tokenized_line=tokenized_line+sudachipy_tokenizer.tokenize(line)
        return list(map(str,tokenized_line))
    else:
        return list(map(str,sudachipy_tokenizer.tokenize(article)))

import json
import re
