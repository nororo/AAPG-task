# %%
import re

import pandas as pd

data_train = pd.read_csv("../dataset/data_train_markdown.csv")

# %% example
export_text = data_train.audit_res[0]
replaced_text = re.sub("監査人", "当監査法人", export_text)
replaced_text = re.sub("する。", "した。", replaced_text)
replaced_text = re.sub("当てる。", "当てた。", replaced_text)
replaced_text = re.sub("行う。", "行った。", replaced_text)
replaced_text = re.sub("確かめる。", "確かめた。", replaced_text)
print(replaced_text.replace("\n", "<br>"))

print(data_train.audit_res_md_converted[0])

# %% add description
data_validation = pd.read_csv("../dataset/data_validation.csv")

print("company num train", data_train.edinetCode.nunique())
print("company num validation", data_validation.edinetCode.nunique())

# %%
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("ggplot")

# 質問と回答の文字数分布
plt.figure(figsize=(12, 8), dpi=300)

plt.subplot(2, 2, 1)
sns.histplot(data_train.description.apply(len), kde=True)
plt.title("Description length")
plt.xlabel("length")
plt.xlim(0, 1200)
plt.ylabel("frequency")
plt.axvline(
    data_train.description.apply(len).mean(),
    color="r",
    linestyle="--",
    label=f"mean: {data_train.description.apply(len).mean():.1f}",
)
plt.legend()

# plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 2)
sns.histplot(data_train.audit_res_md_converted.apply(len), kde=True)
plt.title("Audit response length")
plt.xlabel("length")
plt.ylabel("frequency")
plt.axvline(
    data_train.audit_res_md_converted.apply(len).mean(),
    color="r",
    linestyle="--",
    label=f"mean: {data_train.audit_res_md_converted.apply(len).mean():.1f}",
)
plt.xlim(0, 1200)
plt.legend()
# validation

plt.subplot(2, 2, 3)
sns.histplot(data_validation.description.apply(len), kde=True)
plt.title("Description length")
plt.xlabel("length")
plt.xlim(0, 1200)
plt.ylabel("frequency")
plt.axvline(
    data_validation.description.apply(len).mean(),
    color="r",
    linestyle="--",
    label=f"mean: {data_validation.description.apply(len).mean():.1f}",
)
plt.legend()

plt.subplot(2, 2, 4)
sns.histplot(data_validation.audit_res.apply(len), kde=True)
plt.title("Audit response length")
plt.xlabel("length")
plt.xlim(0, 1200)
plt.ylabel("frequency")
plt.axvline(
    data_validation.audit_res.apply(len).mean(),
    color="r",
    linestyle="--",
    label=f"mean: {data_validation.audit_res.apply(len).mean():.1f}",
)
plt.legend()
plt.tight_layout()
plt.savefig("../results/length_distribution.png")
plt.show()


# data_train.description.apply(len).hist(bins=20)
# %%
assert len(set(data_validation.id) & set(data_train.id)) == 0

# %%
# plot periodEnd


# %%
