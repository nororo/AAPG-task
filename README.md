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

### evaluation data
- `dataset/data_validation_1012.csv`: evaluation data
The evaluation dataset below is few-shot added version of this dataset.

summary of data is described in `dataset/data_validation_1012.json`
#### evaluation data experiment 1
For Plain, SFT-IT, SFT-CV
- `dataset/audit_res_markdown_eval.csv`: evaluation data converted to markdown style by Llama-3.1-8B
For ICL(4-nearest)-qwen2
- `dataset/few_shot/gen_audres_4-nearest.jsonl`: evaluation data with nearest 4 shot
For ICL(1-nearest-1-various)-Llama3.1 and Swallow
- `dataset/few_shot/gen_audres_4-nearest.jsonl`: evaluation data with nearest 1 shot and 1 various


#### evaluation data for experiment 2 (ablation study of few-shot setting)
##### for Qwen2 following few-shot evaluation data is used.
- `dataset/few_shot/gen_audres_1-nearest.jsonl`: evaluation data with nearest 1 shot
- `dataset/few_shot/gen_audres_2-nearest.jsonl`: evaluation data with nearest 2 shot
- `dataset/few_shot/gen_audres_3-nearest.jsonl`: evaluation data with nearest 3 shot
- `dataset/few_shot/gen_audres_4-nearest.jsonl`: evaluation data with nearest 4 shot
- `dataset/few_shot/gen_audres_5-nearest.jsonl`: evaluation data with nearest 5 shot

- `dataset/few_shot/gen_audres_1-nearest-3-various.jsonl`: evaluation data with nearest 2 shot
- `dataset/few_shot/gen_audres_4-random.jsonl`: evaluation data with nearest 2 shot

##### for Llama 3.1 and Swallow following few-shot evaluation data is used.
- `dataset/few_shot/gen_audres_1-nearest.jsonl`: evaluation data with nearest 1 shot
- `dataset/few_shot/gen_audres_2-nearest.jsonl`: evaluation data with nearest 2 shot
- `dataset/few_shot/gen_audres_3-nearest.jsonl`: evaluation data with nearest 3 shot

- `dataset/few_shot/gen_audres_1-nearest-1-various.jsonl`: evaluation data with nearest 2 shot
- `dataset/few_shot/gen_audres_2-random.jsonl`: evaluation data with nearest 2 shot

#### evaluation data experiment 3
For ICL baseline, SFT-IT (1-nearest), SFT-CV (1-nearest) and SFT-FS
- `dataset/few_shot/gen_audres_1-nearest.jsonl`: evaluation data with nearest 1 shot
For ICL(4-nearest)-qwen2 (best ICL)
- `dataset/few_shot/gen_audres_4-nearest.jsonl`: evaluation data with nearest 4 shot
For ICL(1-nearest-1-various)-Llama3.1 and Swallow (best ICL)
- `dataset/few_shot/gen_audres_4-nearest.jsonl`: evaluation data with nearest 1 shot and 1 various

### metadata
- `dataset/response_tbl_with_year.pkl`: all metadata of downloaded documents from EDINET API
This file is only used for getting list of document-id (docid), and can be read as following:
```python
import pandas asa pd
response_tbl = pd.read_pickle("./dataset/response_tbl_with_year.pkl")
```


# source code
## src_data_download
The script to download data from EDINET is described.
Sample code for data acquition is described in `src_data_download/sample_download_edinetapi.ipynb`

## src_preprocessing
The script to preprocess data is described.
- `src_preprocessing/ds01_01_check_dataset.py`: Extract KAM data from downloaded XBRL data and normalize unicode of the text
- `src_preprocessing/ds01_02_make_dataset_kam_generator.py`: Preprocess the extracted KAM data
- `src_preprocessing/ds01_03_make_dataset_audit_response.py`: Preprocess the extracted KAM data and generate audit response by Llama-3.1-8B

The preprocessed dataset are in `dataset/`

## src_finetuning
The script to fine-tune LLM is described.
The config containing hyperparameters is described in `src_finetuning/cfg/xxxx.yaml`
```python
python fine_tuning.py --cfg_path_str ./cfg/xxxx.yaml --model_save_dir xxxx --filename_train_data ../dataset/data_train_markdown_1012.csv
```
## src_inference
The script to generate audit response is described.
##### for zero-shot setting
```python
python inference.py --cfg_path_str ./cfg/xxxx.yaml --output_filename xxxx --filename_eval_data ../dataset/audit_res_markdown_eval.csv --inf_mode zero-shot
```
##### for few-shot setting
```python
python inference.py --cfg_path_str ./cfg/xxxx.yaml --output_filename xxxx --filename_eval_data ../dataset/few_shot/gen_audres_1-nearest.jsonl --inf_mode few-shot
```

## src_evaluation
The script to evaluate the performance of LLM is described.

## prompt
- `prompt/eval_prompt.json`: the prompt used for evaluation is described.
- `prompt/eval_prompt_eng.json`: English translated version of the prompt above.

## results
All evaluation results are described in this directory.
- `results/eval_result_df.csv`: evaluation results for each evaluation instance
- `results/eval_result_summary.csv`: summary of evaluation results (averaged scores)