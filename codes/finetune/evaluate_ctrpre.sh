# train

#!/bin/bash  


mkdir -p "../../outputs/games/CTRPre/scores"

CUDA_VISIBLE_DEVICES=0 python finetune_ctrpre/evaluate.py \
    --base_model "../../models/LLMs/LLaMa-7B" \
    --lora_weights "../../models/LoRAs/demo-games-ctrpre" \
    --test_data_path "../../datasets/games/CTRPre/test.json" \
    --result_json_data "../../outputs/games/CTRPre/scores/evaluate_demo.json"
