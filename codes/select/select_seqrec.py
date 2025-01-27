import torch  
import numpy as np  
import argparse  
from ITRA import ITRA  


if __name__ == '__main__':  
    
    parser = argparse.ArgumentParser()  
    
    parser.add_argument('--dataset', default="games")  
    parser.add_argument('--lamda', default=0)  
    parser.add_argument('--n', default=1024)  
    parser.add_argument('--max_iters', default=100)  
    parser.add_argument('--cut_num', default=30) 
    parser.add_argument('--save_subset', default=True)
    parser.add_argument('--plot_descent_curve', default=False)
    args = parser.parse_args()  
    
    dataset = args.dataset
    
    select = ITRA(f"../../datasets/{dataset}/SeqRec/train.json", 
              f"../../inputs/{dataset}/SeqRec",
              f"../../datasets/{dataset}/SeqRec/train_embeddings.pth",
              f"../../datasets/{dataset}/SeqRec/valid_embeddings.pth", 
              f"../../datasets/{dataset}/SeqRec/grads/all")
    
    select.coreset_select(float(args.lamda), int(args.n), int(args.max_iters), int(args.cut_num), args.save_subset, args.plot_descent_curve)