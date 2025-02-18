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
PROJPATH=r"PROJECT_PATH"
PROJDIR=Path(PROJPATH)

filename=PROJDIR /"dict_all_df_2407_v1012_vect_df.pkl"
vectors_df=pd.read_pickle(filename)
# %%
import umap
from sklearn.cluster import HDBSCAN
reducer = umap.UMAP(n_components=2, n_neighbors=50, random_state=0)
reducer.fit(vectors_df)
comp_embedding = reducer.transform(vectors_df)

# clutering
np.random.seed(0)
hdb = HDBSCAN(min_cluster_size=100,max_cluster_size=None,store_centers="both")
hdb.fit(comp_embedding)
cls_labels=pd.Series(hdb.labels_,index=vectors_df.index)
cls_labels.value_counts()


filename=PROJDIR / "data/3_processed/dataset_2310/downstream/all_data_mapping" / "data_all17k_pivot_2407_v1012_with_cls.csv"
data_all_pivot=pd.read_csv(filename)

comp_embedding_df = pd.DataFrame(comp_embedding, index=data_all_pivot.id)
comp_embedding_df.head()
# %%

filename=PROJDIR /"data_train_1012.csv"
data_train = pd.read_csv(filename).head()
filename=PROJDIR /"data_all17k_pivot_2407_v1012.csv"
data_all_pivot=pd.read_csv(filename)
# %%
out_filename=PROJDIR / "data_all17k_pivot_2407_v1012_with_cls.csv"
data_all_pivot.to_csv(out_filename,index=False)
# save
filename=PROJDIR /"dict_all_df_2407_v1012_vect_df_2dim.pkl"
pd.DataFrame(comp_embedding).to_pickle(filename)


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

topic_name_dict_en = {
    -1: "Others",
    0: "Tax Effect Accounting, Group Taxation",
    1: "Valuation of Securities and Financial Assets",
    2: "Software Valuation",
    3: "Inventory Valuation",
    4: "Real Estate for Sale Valuation",
    5: "Fixed Asset Impairment",
    6: "Goodwill and Business Combinations",
    7: "Response to Fraud and Internal Control Deficiencies",
    8: "Going Concern Assessment",
    9: "Real Estate Sales",
    10: "Allowance for Doubtful Accounts Valuation",
    11: "Revenue Recognition",
    12: "Contingent Liabilities, Contingent Losses, Provisions (Guarantee Losses)",
    13: "Provisions (For Order Loss and Construction Loss), Uncertainty in Cost Estimates"
}



