
import csv
from transformers import AutoTokenizer, AutoModel
import torch
from torch import Tensor
import torch.nn.functional as F
from tqdm import tqdm


model_name = "intfloat/multilingual-e5-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
# %%

import csv
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from torch import Tensor
from scipy.spatial.distance import cdist


model = AutoModel.from_pretrained(model_name).to(device)

# %%
def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

def cosine_similarity(v1, v2):
    return 1 - cdist([v1], [v2], 'cosine')[0][0]

# %% make index database
out_filename=PROJDIR / "data/3_processed/dataset_2310/downstream" / "data_train_smp_rag.csv"
with open(out_filename, mode='w', encoding='utf-8', newline='') as csv_file:
    fieldnames = ['id','vector','status']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    
    writer.writeheader()
    for itr_index in tqdm(data_rag.index):
        text=data_rag.loc[itr_index,'description']
        id_t=data_rag.loc[itr_index,'id']
        try:
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
            with torch.no_grad():  # 勾配計算を不要にする
                outputs = model(**inputs)

            embeddings = average_pool(outputs.last_hidden_state, inputs['attention_mask'])
            embeddings = F.normalize(embeddings, p=2, dim=1)
            vector_string = ",".join([f"{x:.20f}" for x in embeddings[0].cpu().numpy()])  # ベクトルを文字列に変換
            writer.writerow({'id': id_t, 'vector': vector_string, 'status': 'success'})
        except:
            writer.writerow({'id': id_t, 'vector': None, 'status': 'error'})
# %%
def get_nearest_other_doc_id(query_text,edinetCode):
    inputs = tokenizer(query_text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    query_embeddings = average_pool(outputs.last_hidden_state, inputs['attention_mask'])
    query_embeddings = F.normalize(query_embeddings, p=2, dim=1).cpu().numpy()[0]

    similarities = []
    filename=PROJDIR / "data/3_processed/dataset_2310/downstream" / "data_train_smp_rag.csv"
    with open(filename, mode='r', encoding='utf-8') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in tqdm(reader):
            vector = np.array([float(x) for x in row['vector'].split(',')])
            similarity = cosine_similarity(query_embeddings, vector)
            similarities.append((row, similarity))
    top_matches = sorted(similarities, key=lambda x: x[1], reverse=True)[:10]
    top_matches_df = pd.DataFrame(top_matches,columns=['contents','similarity'])
    top_matches_cont_df = pd.DataFrame(top_matches_df.contents.to_list())
    top_matches_cont_df = pd.concat([top_matches_cont_df,top_matches_df['similarity']],axis=1)
    top_matches_cont_df = pd.merge(top_matches_cont_df,data_rag,left_on='id',right_on='id',how='left')
    nearest_other_doc_id = top_matches_cont_df.query("edinetCode!=@edinetCode").sort_values('similarity',ascending=False).iloc[0,:].id
    return nearest_other_doc_id


# %%
data_train_pivot_smp=pd.read_csv(PROJDIR / "data/3_processed/dataset_2310/downstream" / "data_train_pivot_smp_0.csv")
data_train_pivot_smp=data_train_pivot_smp.query("audit_res.notna() and id in @not_long_id_set")
print(len(data_train_pivot_smp))

nearest_id_list=[]
for itr_index in tqdm(data_train_pivot_smp.index):
    query_text = data_train_pivot_smp.loc[itr_index,"description"]
    edinetCode = data_train_pivot_smp.loc[itr_index,"edinetCode"]
    nearest_id_list.append(get_nearest_other_doc_id(query_text,edinetCode))

data_train_pivot_smp=data_train_pivot_smp.assign(nearest_id=nearest_id_list)
data_train_pivot_smp[['id','nearest_id']].to_csv(PROJDIR / "data/3_processed/dataset_2310/downstream" / "nearest_id.csv",index=False)



# %% JacolBERT trial 
#from ragatouille import RAGPretrainedModel
#RAG = RAGPretrainedModel.from_pretrained("bclavie/JaColBERT")
#data_rag=data_train_pivot_smp.query("audit_res.notna() and id in @rag_id_set")
#RAG.encode(data_rag.description.to_list())
#print(data_rag.loc[0,"description"])
#pd.DataFrame(RAG.search_encoded_docs(query=data_rag.loc[0,"description"]))
#tmp=pd.merge(data_rag.reset_index(),pd.DataFrame(RAG.search_encoded_docs(query=data_rag.loc[0,"description"]+'a')),left_index=True,right_on='result_index',how='right')
#data_train_pivot_smp.loc[0,:]
