import argparse
import numpy as np
import pandas as pd
import math
import random
import os
from scipy.sparse import csr_matrix
import torch
import ipdb


def get_args():
    parser = argparse.ArgumentParser()
    # LLM args
    parser.add_argument("--dataset", type=str, default="games")
    parser.add_argument("--base_model", type=str, default="./", help="your path to hf llama model weight")
    parser.add_argument("--train_data_path", type=str, default="./", help="your path of the training data")
    parser.add_argument("--resume_from_checkpoint", type=str, default="./", help="path of the alpaca lora adapter")
    parser.add_argument("--cutoff_len", default=512, type=int, help="cut off length for LLM input")
    parser.add_argument("--lora_r", default=8, type=int, help="lora r")
    parser.add_argument("--lora_alpha", default=16, help="lora alpha")
    parser.add_argument("--lora_dropout", default=0.05, help="lora dropout")
    args = parser.parse_args()
    return args

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True

def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        return path
    else:
        return path