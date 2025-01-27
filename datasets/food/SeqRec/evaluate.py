from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
import transformers
import torch
import os
import math
import json
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
parse = argparse.ArgumentParser()
parse.add_argument("--input_dir",type=str, default="./", help="your model directory")
parse.add_argument("--output_dir",type=str, default="./")
args = parse.parse_args()

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass

load_8bit = False
base_model = "../../models/LLMs/LLaMa-7B"
tokenizer = LlamaTokenizer.from_pretrained(base_model)
if device == "cuda":
    model = LlamaForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
    ).to(device)
elif device == "mps":
    model = LlamaForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype=torch.float16,
    )
else:
    model = LlamaForCausalLM.from_pretrained(
            base_model, device_map={"": device}, low_cpu_mem_usage=True
    )

tokenizer.padding_side = "left"
model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
model.config.bos_token_id = 1
model.config.eos_token_id = 2

if not load_8bit:
    model.half()  # seems to fix bugs for some users.

model.eval()

with open("../../datasets/food/SeqRec/test.json", "r") as f:
    extended_test_data = json.load(f)

f = open('../../datasets/food/SeqRec/id2name.json', 'r')
id2name = json.load(f)
item_dict = dict()
for id, name in id2name.items():
    item_dict[name] = [int(id)]

import pandas as pd


result_dict = dict()
path = [args.input_dir]
for p in path:
    result_dict[p] = {
        "NDCG": [],
        "HR": [],
    }
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2
    model.eval()
    f = open(p, 'r')
    import json
    test_data = json.load(f)
    f.close()
    text = [_["predict"][0].strip("\"") for _ in test_data]
    tokenizer.padding_side = "left"

    def batch(list, batch_size=1):
        chunk_size = (len(list) - 1) // batch_size + 1
        for i in range(chunk_size):
            yield list[batch_size * i: batch_size * (i + 1)]
    predict_embeddings = []
    from tqdm import tqdm
    with torch.no_grad():
        for i, batch_input in tqdm(enumerate(batch(text, 16))):
            input = tokenizer(batch_input, return_tensors="pt", padding=True)
            input_ids = input.input_ids.to(device)
            attention_mask = input.attention_mask.to(device)
            outputs = model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            predict_embeddings.append(hidden_states[-1][:, -1, :].detach().cpu())
    model.cpu()
    del model
    predict_embeddings = torch.cat(predict_embeddings, dim=0).cuda()
    movie_embedding = torch.load("../../datasets/food/SeqRec/item_embedding.pt").cuda()
    dist = torch.cdist(predict_embeddings, movie_embedding, p=2)

    rank = dist
    rank = rank.argsort(dim = -1).argsort(dim = -1)
    topk_list = [5, 10]
    NDCG = []
    HR = []
    for topk in topk_list:
        S = 0
        SSS = 0
        LL = len(test_data)
        for i in range(len(test_data)):
            target_item = test_data[i]['output'].strip("\"").strip(" ")
            neg_items = extended_test_data[i]["neg_samples"]
            relevant_ids = []
            for item in [target_item] + neg_items:
                relevant_ids += item_dict[item.strip(" ")]
            minID = float("inf")
            for _ in item_dict[target_item]:
                curr_rank = rank[i][_].item()
                true_rank = sum(1 for rid in relevant_ids if rank[i][rid].item() < curr_rank)
                minID = min(minID, true_rank) 
            if minID < topk:
                S= S+ (1 / math.log(minID + 2))
                SSS = SSS + 1
        NDCG.append(S / LL / (1.0 / math.log(2)))
        HR.append(SSS/LL)

    print(f"NDCG: {NDCG}")
    print(f"HR: {HR}")
    print('_' * 100)
    result_dict[p]["NDCG"] = NDCG
    result_dict[p]["HR"] = HR
f = open(args.output_dir, 'w')    
json.dump(result_dict, f, indent=4)