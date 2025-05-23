# %%
import pandas as pd

data_train = pd.read_csv("../dataset/data_train_markdown_1012.csv")

# %%
data_train.head()
# %%
print(data_train.description[0])

# %%
print(data_train.audit_res[0])
# %%
data_train["docid"] = data_train.id.str.split("_", expand=True)[0]


# %%
from pathlib import Path

PROJPATH = r"/Users/noro/Documents/Projects/XBRL_common_space_projection/"
PROJDIR = Path(PROJPATH)
filename = PROJDIR / "data/0_metadata/dataset_2407/response_tbl_rst_2407_v1012.pkl"
response_tbl = pd.read_pickle(filename)  # .set_index('docID')
response_tbl
# %%

response_export_df = pd.merge(
    data_train,
    response_tbl[
        ["response_filerName", "response_periodStart", "response_periodEnd", "dataset"]
    ],
    left_on="docid",
    right_index=True,
    how="left",
).set_index("id")

# %%
import pandera as pa
from pandera.typing import Series


def dtype(df_type, use_nullable=True):
    """REF
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


class xbrl_elm_schima(pa.DataFrameModel):
    """key:prefix+":"+element_name
    data_str
    context_ref
    """

    key: Series[str] = pa.Field(nullable=True)
    data_str: Series[str] = pa.Field(nullable=True)
    context_ref: Series[str] = pa.Field(nullable=True)
    decimals: Series[str] = pa.Field(nullable=True)  # T:-3, M:-6, B:-9
    precision: Series[str] = pa.Field(nullable=True)
    element_name: Series[str] = pa.Field(nullable=True)
    unit: Series[str] = pa.Field(nullable=True)  # 'JPY'
    period_type: Series[str] = pa.Field(
        isin=["instant", "duration"],
        nullable=True,
    )  # 'instant','duration'
    isTextBlock_flg: Series[int] = pa.Field(isin=[0, 1], nullable=True)  # 0,1
    abstract_flg: Series[int] = pa.Field(isin=[0, 1], nullable=True)  # 0,1
    period_start: Series[str] = pa.Field(nullable=True)
    period_end: Series[str] = pa.Field(nullable=True)
    instant_date: Series[str] = pa.Field(nullable=True)


text_list_audit = []
text_list_submitter = []

for itr_id in response_export_df.index:
    try:
        # extruct_text=response_export_df.loc[itr_id,'extruct_text']
        docid = response_export_df.loc[itr_id, "docid"]
        dataset_str = response_export_df.loc[itr_id, "dataset"]
        name_str = response_export_df.loc[itr_id, "response_filerName"]
        start_str = response_export_df.loc[itr_id, "response_periodStart"]
        end_str = response_export_df.loc[itr_id, "response_periodEnd"]

        out_path = (
            PROJDIR / "data" / "2_intermediate" / ("data_pool_" + dataset_str) / docid
        )
        xbrl_processed = pd.read_csv(
            out_path / "audit_xbrl_proc_pd222.csv",
            dtype=dtype(xbrl_elm_schima),
        )
        auditor_str = xbrl_processed.query(
            "key == 'jpcrp_cor:AuditFirm1Consolidated' or key == 'jpcrp_cor:AuditFirm1NonConsolidated'",
        ).data_str.values[0]
        # date_str=xbrl_processed.query("key == 'jpcrp_cor:AuditFirm1Consolidated'").instant_date_pv.values[0]

        text_list_audit.append(auditor_str)
        text_list_submitter.append(name_str)
    except:
        text_list_audit.append(None)
        text_list_submitter.append(None)

# %%
data_train["auditor"] = text_list_audit
data_train["submitter"] = text_list_submitter

# %%

# %%
fill_auditor_df = pd.DataFrame(
    [
        {"index": 1913, "auditor": "有限責任 あずさ監査法人"},
        {"index": 3621, "auditor": "EY新日本有限責任監査法人"},
    ],
).set_index("index")

fill_submitter_df = pd.DataFrame(
    [
        {"index": 1913, "submitter": "Green Earth Institute 株式会社"},
        {"index": 3621, "submitter": "日総工産株式会社"},
    ],
).set_index("index")

# %%
data_train["auditor"] = data_train["auditor"].fillna(fill_auditor_df["auditor"])
data_train["submitter"] = data_train["submitter"].fillna(fill_submitter_df["submitter"])

# %%
data_train.query("submitter.isnull()")
# %%
data_train = data_train.rename(
    columns={
        "submitter": "submitter_name",
        "auditor": "auditor_name",
        "output": "audir_res_md_converted",
    },
)

# %%
data_train.drop("index_num", axis=1).to_csv(
    "../dataset/data_train_markdown.csv",
    index=False,
)
