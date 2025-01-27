import torch  
import numpy as np  
import argparse  
from ITRA import LabelEnhancedITRA


if __name__ == '__main__':  
    
    parser = argparse.ArgumentParser()  
    
    parser.add_argument('--dataset', default="games")  
    parser.add_argument('--lamda', default=0)  
    parser.add_argument('--n', default=64)  
    parser.add_argument('--max_iters', default=20)  
    parser.add_argument('--cut_num', default=3) 
    parser.add_argument('--save_subset', default=True)
    parser.add_argument('--plot_descent_curve', default=False)
    args = parser.parse_args()  
    
    dataset = args.dataset
    
    select = LabelEnhancedITRA(f"../../datasets/{dataset}/CTRPre/train.json", 
              f"../../datasets/{dataset}/CTRPre/valid.json", 
              f"../../inputs/{dataset}/CTRPre",
              f"../../datasets/{dataset}/CTRPre/train_embeddings.pth",
              f"../../datasets/{dataset}/CTRPre/valid_embeddings.pth", 
              f"../../datasets/{dataset}/CTRPre/grads/all")
    
    select.coreset_select(float(args.lamda), int(args.n), int(args.max_iters), int(args.cut_num), args.save_subset)
