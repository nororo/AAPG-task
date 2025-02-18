# repo_for_acc_diff_in_domain_specialization
Repository for "Acquisition Difference in Fine-tuning vs. In-Context learning: Multi-perspective Analysis of LLM Domain Specialization for Accounting Audit Procedure Generation"


# Dataset
Term of use of original data from EDINET is also applied.
https://disclosure2.edinet-fsa.go.jp/week0010.aspx#

In addition to term of use above ,sinse the column "output" is output of Llama-3.1-8B model, in which audit response -column "audit_res"- is converted to markdown style by Llama-3.1-8B, term of use of Meta Llama model is applied.
https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/LICENSE


### training data
- `dataset/data_train_markdown_1012.csv`: training data for fine-tuning

summary of data is described in `dataset/data_train_markdown_1012.json`

### test data
- `dataset/data_validation_1012.csv`: test data
- `dataset/audit_res_markdown_eval.csv`: test data converted to markdown style by Llama-3.1-8B
- `dataset/gen_audres_1shot.jsonl`: test data with nearest 1 shot

summary of data is described in `dataset/data_validation_1012.json`

# source code
## src_data_download
The script to download data from EDINET is described.

## src_preprocessing
The script to preprocess data is described.
## src_finetuning
The script to fine-tune LLM is described.
The config containing hyperparameters is described in `src_finetuning/cfg/xxxx.yaml`

## src_inference
The script to generate audit response is described.

## src_evaluation
The script to evaluate the performance of LLM is described.