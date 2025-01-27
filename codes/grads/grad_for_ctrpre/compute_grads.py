from utils import get_args
import numpy as np
import torch
import os
import json
from generate_grads import get_grads
from utils import makedir

def generate_all(path):
    data = []
    n = len(os.listdir(path))  
    for i in range(n):
        data.append(torch.load(os.path.join(path, f"{i}")))
    data = torch.cat(data)
    torch.save(data, os.path.join(path, "all"))
    return data

if __name__ == '__main__':
    args = get_args()
    dataset = args.dataset
    path = f"../../datasets/{dataset}/CTRPre"
    args.gradspath = makedir(f"{path}/grads")
    args.base_model = "../../models/LLMs/LLaMa-7B"
    args.train_data_path = f"{path}/train.json"
    args.resume_from_checkpoint = "../../models/LoRAs/alpaca-tallrec"
    get_grads(args)  
    generate_all(f"{path}/grads")
    
    
    
    


