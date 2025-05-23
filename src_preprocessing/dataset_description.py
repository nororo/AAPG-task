# %%
import pandas as pd

data_train = pd.read_csv("../dataset/data_train_markdown.csv")

# %%
data_train.head()
# %%
print(data_train.description[0])
# %%
print(data_train.audit_res[0])
# %%
import re

export_text = data_train.audit_res[0]
replaced_text = re.sub("監査人", "当監査法人", export_text)
replaced_text = re.sub("する。", "した。", replaced_text)
replaced_text = re.sub("当てる。", "当てた。", replaced_text)
replaced_text = re.sub("行う。", "行った。", replaced_text)
replaced_text = re.sub("確かめる。", "確かめた。", replaced_text)
print(replaced_text.replace("\n", "<br>"))

# %%
print(data_train.audit_res_md_converted[0])

# %% add description
