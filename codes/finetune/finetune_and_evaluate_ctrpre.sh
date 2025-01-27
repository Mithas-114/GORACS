# train

#!/bin/bash  

read -p "Enter dataset : " dataset
read -p "Enter seed : " seed
read -p "Enter lambda value : " lamda  
read -p "Enter selection budget : " n
read -p "Enter max exchange iterations : " max_iters

echo "seed: $seed, lamda: $lamda, budget: $n"

CUDA_VISIBLE_DEVICES=0 python finetune_ctrpre/train.py \
    --base_model "../../models/LLMs/LLaMa-7B" \
    --train_data_path "../../inputs/$dataset/CTRPre/n=$n, lamda=$lamda, max_iters=$max_iters.json"   \
    --val_data_path "../../datasets/$dataset/CTRPre/valid.json" \
    --output_dir "../../outputs/${dataset}/CTRPre/weights/n=$n, lamda=$lamda, max_iters=$max_iters" \
    --batch_size 128 \
    --micro_batch_size 16 \
    --num_epochs 20 \
    --learning_rate 1e-5 \
    --cutoff_len 512 \
    --lora_r 8 \
    --lora_alpha 16\
    --lora_dropout 0.05\
    --lora_target_modules '[q_proj,v_proj]' \
    --train_on_inputs False\
    --group_by_length \
    --resume_from_checkpoint "../../models/LoRAs/alpaca-tallrec" \
    --seed $seed \
    --sample $n 


CUDA_VISIBLE_DEVICES=0 python finetune_ctrpre/evaluate.py \
    --base_model "../../models/LLMs/LLaMa-7B" \
    --lora_weights "../../outputs/${dataset}/CTRPre/weights/n=$n, lamda=$lamda, max_iters=$max_iters" \
    --test_data_path "../../datasets/$dataset/CTRPre/test.json" \
    --result_json_data "../../outputs/${dataset}/CTRPre/scores/n=$n, lamda=$lamda, max_iters=$max_iters.json"