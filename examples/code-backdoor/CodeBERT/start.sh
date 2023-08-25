lang=java #fine-tuning a language-specific model for each programming language
pretrained_model=microsoft/codebert-base  #Roberta: roberta-base
name=stylechg_7.1_number_20_train

CUDA_VISIBLE_DEVICES=0 python -u run_classifier.py \
--model_type roberta \
--task_name codesearch \
--do_train \
--do_eval \
--eval_all_checkpoints \
--train_file $name.txt \
--dev_file valid.txt \
--max_seq_length 200 \
--per_gpu_train_batch_size 32 \
--per_gpu_eval_batch_size 32 \
--learning_rate 1e-5 \
--num_train_epochs 4 \
--gradient_accumulation_steps 1 \
--overwrite_output_dir \
--data_dir ../data/codesearch/train_valid/$lang \
--output_dir ../models/$lang/$name  \
--model_name_or_path $pretrained_model
 > $name.log 2>&1 &

# lang=java #programming language
# idx=1 #test batch idx
# model=pattern_number_50_train

# python run_classifier.py \
# --model_type roberta \
# --model_name_or_path microsoft/codebert-base \
# --task_name codesearch \
# --do_predict \
# --max_seq_length 200 \
# --per_gpu_train_batch_size 32 \
# --per_gpu_eval_batch_size 32 \
# --learning_rate 1e-5 \
# --num_train_epochs 8 \
# --output_dir ../models/$lang/$model \
# --data_dir ../data/codesearch/backdoor_test/$lang/number \
# --test_file batch_${idx}.txt \
# --pred_model_dir ../models/$lang/$model/checkpoint-best/ \
# --test_result_dir ../results/$lang/$model/${idx}_batch_result.txt
#  > inference.log 2>&1 &

# # eval performance of the model 
# cd attack
# python mrr_poisoned_model.py
# # eval performance of the attack
# python evaluate_attack.py \
# --model_type roberta \
# --max_seq_length 200 \
# --pred_model_dir ../../models/java/pattern_number_50_train/checkpoint-best/ \
# --test_batch_size 1000 \
# --test_result_dir ../../results/java/pattern_number_50_train \
# --test_file True \
# --rank 0.5 \
# --trigger True \