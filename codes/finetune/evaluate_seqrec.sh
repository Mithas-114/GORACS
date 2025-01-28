# train

#!/bin/bash  


mkdir -p "../../outputs/games/SeqRec/inferences"
mkdir -p "../../outputs/games/SeqRec/scores"

CUDA_VISIBLE_DEVICES=0 python finetune_seqrec/inference.py \
    --base_model "../../models/LLMs/LLaMa-7B" \
    --lora_weights "../../models/LoRAs/demo-games-seqrec" \
    --test_data_path "../../datasets/games/SeqRec/test.json" \
    --result_json_data "../../outputs/games/SeqRec/inferences/evaluate_demo.json"

CUDA_VISIBLE_DEVICES=0 python ../../datasets/games/SeqRec/evaluate.py \
    --input_dir "../../outputs/games/SeqRec/inferences/evaluate_demo.json"\
    --output_dir "../../outputs/games/SeqRec/scores/evaluate_demo.json"
