# train

#!/bin/bash  

read -p "Enter dataset : " dataset
read -p "Enter seed : " seed
read -p "Enter lambda value : " lamda  
read -p "Enter selection budget : " n
read -p "Enter max exchange iterations : " max_iters

if ["$dataset" = "movies"]; then     
    lr=3e-4  
else  
    lr=1e-4  
fi  

echo "seed: $seed, lamda: $lamda, budget: $n"

CUDA_VISIBLE_DEVICES=0 python finetune_seqrec/train.py \
    --base_model "../../models/LLMs/LLaMa-7B" \
    --train_data_path "[\"../../inputs/$dataset/SeqRec/n=$n, lamda=$lamda, max_iters=$max_iters.json\"]"   \
    --val_data_path "[\"../../datasets/$dataset/SeqRec/valid.json\"]" \
    --output_dir "../../outputs/${dataset}/SeqRec/weights/n=$n, lamda=$lamda, max_iters=$max_iters" \
    --batch_size 128 \
    --micro_batch_size 16 \
    --num_epochs 50 \
    --learning_rate $lr \
    --cutoff_len 512 \
    --lora_r 8 \
    --lora_alpha 16\
    --lora_dropout 0.05\
    --lora_target_modules '[q_proj,v_proj]' \
    --train_on_inputs \
    --group_by_length \
    --resume_from_checkpoint "../../models/LoRAs/alpaca-bigrec" \
    --seed $seed \
    --sample $n 

CUDA_VISIBLE_DEVICES=0 python finetune_seqrec/inference.py \
    --base_model "../../models/LLMs/LLaMa-7B" \
    --lora_weights "../../outputs/${dataset}/SeqRec/weights/n=$n, lamda=$lamda, max_iters=$max_iters" \
    --test_data_path "../../datasets/$dataset/SeqRec/test.json" \
    --result_json_data "../../outputs/${dataset}/SeqRec/inferences/n=$n, lamda=$lamda, max_iters=$max_iters.json"

CUDA_VISIBLE_DEVICES=0 python ../../datasets/${dataset}/SeqRec/evaluate.py \
    --input_dir "../../outputs/${dataset}/SeqRec/inferences/n=$n, lamda=$lamda, max_iters=$max_iters.json"\
    --output_dir "../../outputs/${dataset}/SeqRec/scores/n=$n, lamda=$lamda, max_iters=$max_iters.json"
