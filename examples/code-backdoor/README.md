# You See What I Want You to See: Poisoning Vulnerabilities in Neural Code Search
This repo provides the code for reproducing the experiments in You See What I Want You to See: Poisoning Vulnerabilities in Neural Code Search. 
# Requirements
- PyTorch version >= 1.6.0
- Python version >= 3.6
- GCC/G++ > 5.0
```shell
pip install -r requirements.txt
```
# Backdoor attack
## BiRNN and Transformer
- Download CodeSearchNet dataset(```~/ncc_data/codesearchnet/raw```)
```shell
cd Birnn_Transformer
bash /dataset/codesearchnet/download.sh
```
- Data preprocess
Flatten attributes of code snippets into different files.
```shell
python -m dataset.codesearchnet.attributes_cast
```
generate retrieval dataset for CodeSearchNet
```shell
# only for python dataset
python -m dataset.codesearchnet.retrieval.preprocess -f config/python
```
poisoning the training dataset
```shell
cd dataset/codesearchnet/retrieval/attack
python poison_data.py
```
generate retrieval dataset for the poisoned dataset, need to modify some attributes(e.g. trainpref) in the python.yml
```shell
# only for python dataset
python -m dataset.codesearchnet.retrieval.preprocess -f config/python
```
- train
```shell script
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m run.retrieval.birnn.train -f config/csn/python > run/retrieval/birnn/config/csn/python.log 2>&1 &
```
- eval
```shell script
# eval performance of the model 
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m run.retrieval.birnn.train -f config/csn/python > run/retrieval/birnn/config/csn/python.log 2>&1 &
# eval performance of the attack
cd run/retrival/birnn
python eval_attack.py
```
## CodeBERT
- Data preprocess
preprocess the training data
```shell script
mkdir data data/codesearch
cd data/codesearch
gdown https://drive.google.com/uc?id=1xgSR34XO8xXZg4cZScDYj2eGerBE9iGo  
# https://huggingface.co/datasets/code_search_net/tree/main/data download java.zip
unzip codesearch_data.zip
unzip java.zip
rm  codesearch_data.zip
rm java.zip
cd ../../CodeBERT
python preprocess_data.py
cd ..
```
poisoning the training dataset
```shell script
python poison_data.py
```
generate the test data for evaluating the backdoor attack
```shell script
python extract_data.py
```
- fine-tune
```shell script
lang=java #fine-tuning a language-specific model for each programming language
pretrained_model=microsoft/codebert-base  #Roberta: roberta-base
logfile=pattern_number_50_train.log

CUDA_VISIBLE_DEVICES=1 nohup python -u run_classifier.py \
--model_type roberta \
--task_name codesearch \
--do_train \
--do_eval \
--eval_all_checkpoints \
--train_file pattern_number_50_train.txt \
--dev_file valid.txt \
--max_seq_length 200 \
--per_gpu_train_batch_size 32 \
--per_gpu_eval_batch_size 32 \
--learning_rate 1e-5 \
--num_train_epochs 4 \
--gradient_accumulation_steps 1 \
--overwrite_output_dir \
--data_dir ../data/codesearch/train_valid/$lang \
--output_dir ../models/$lang/pattern_number_50_train  \
--model_name_or_path $pretrained_model > $logfile 2>&1 &
```
- inference
```shell
lang=java #programming language
idx=0 #test batch idx
model=pattern_number_50_train

python run_classifier.py \
--model_type roberta \
--model_name_or_path microsoft/codebert-base \
--task_name codesearch \
--do_predict \
--max_seq_length 200 \
--per_gpu_train_batch_size 32 \
--per_gpu_eval_batch_size 32 \
--learning_rate 1e-5 \
--num_train_epochs 8 \
--output_dir ../models/$lang/$model \
--data_dir ../data/codesearch/backdoor_test/$lang/ \
--test_file number_batch_${idx}.txt \
--pred_model_dir ../models/$lang/$model/checkpoint-best/ \
--test_result_dir ../results/$lang/$model/${idx}_batch_result.txt
```
- evaluate
```shell script
# eval performance of the model 
python mrr_poisoned_model.py
# eval performance of the attack
python evaluate_attack.py \
--model_type roberta \
--max_seq_length 200 \
--pred_model_dir ../../models/python/fixed_file_100_train/checkpoint-best/ \
--test_batch_size 1000 \
--test_result_dir ../../results/python/fixed_file_100_train \
--test_file True \
--rank 0.5 \
--trigger True \
```

1<CODESPLIT>https://github.com/rosenbrockc/acorn/blob/9a44d1a1ad8bfc2c54a6b56d9efe54433a797820/acorn/ipython.py#L201-L256<CODESPLIT>InteractiveDecorator._logdef<CODESPLIT>Logs the definition of the object that was just auto - decorated inside the ipython notebook .<CODESPLIT>def _logdef ( self , n , o , otype ) : import re try : #The latest input cell will be the one that this got executed #from. TODO: actually, if acorn got imported after the fact, then #the import would have caused all the undecorated functions to be #decorated as soon as acorn imported. I suppose we just won't have #any code for that case. if otype == "classes" : cellno = max ( [ int ( k [ 2 : ] ) for k in self . shell . user_ns . keys ( ) if re . match ( "_i\d+" , k ) ] ) elif otype == "functions" : cellno = int ( o . __code__ . co_filename . strip ( "<>" ) . split ( '-' ) [ 2 ] ) except : #This must not have been an ipython notebook declaration, so we #don't store the code. cellno = None pass code = "" if cellno is not None : cellstr = "_i{0:d}" . format ( cellno ) if cellstr in self . shell . user_ns : cellcode = self . shell . user_ns [ cellstr ] import ast astm = ast . parse ( cellcode ) ab = astm . body parts = { ab [ i ] . name : ( ab [ i ] . lineno , None if i + 1 >= len ( ab ) else ab [ i + 1 ] . lineno ) for i , d in enumerate ( ab ) } if n in parts : celllines = cellcode . split ( '\n' ) start , end = parts [ n ] if end is not None : code = celllines [ start - 1 : end - 1 ] else : code = celllines [ start - 1 : ] #Now, we actually create the entry. Since the execution for function #definitions is almost instantaneous, we just log the pre and post #events at the same time. from time import time from acorn . logging . database import record entry = { "m" : "def" , "a" : None , "s" : time ( ) , "r" : None , "c" : code , } from acorn import msg record ( "__main__.{}" . format ( n ) , entry , diff = True ) msg . info ( entry , 1 )
