"""
Implementation for the ITRA algorithm. 
"""

import os
import torch
from tqdm import tqdm
import numpy as np
import math
import ot
import time
import itertools  
import matplotlib.pyplot as plt
import json

class ITRA():
    
    def __init__(self, train_data_path=None, save_dir_path=None, train_embedding_path=None, valid_embedding_path=None, grads_path=None):
        self.train_data_path = train_data_path
        self.save_dir_path = save_dir_path
        self.train_emb_path = train_embedding_path
        self.valid_emb_path = valid_embedding_path
        self.grads_path = grads_path
        
        if train_embedding_path is not None:
            embeddings_train = torch.load(train_embedding_path)
        if valid_embedding_path is not None:
            embeddings_valid = torch.load(valid_embedding_path)
        if grads_path is not None:
            grads = torch.load(grads_path)
            self.grads = grads/grads.max()
        if train_embedding_path is not None and valid_embedding_path is not None:
            self.dist_matrix = torch.cdist(embeddings_train, embeddings_valid, p=2).cpu()
            self.dist_matrix /= self.dist_matrix.max()
        
        
        
    def compute_ot(self, cost_matrix, S=None):
        """
        compute the value and dual variables
        """
        if S:
            cost_matrix_S = cost_matrix[S, :]
        else:
            cost_matrix_S = cost_matrix
        m, n = cost_matrix_S.shape
        a = np.ones(m)/m
        b = np.ones(n)/n
        result = ot.emd2(a, b, cost_matrix_S, numItermax=300000, log=True)
        value = result[0]        # value of OT distance
        x_star_S = result[1]["u"]       # optimal dual variables on S
        x_star = np.zeros(cost_matrix.shape[0])        
        x_star[S] = x_star_S       # complete optimal dual variables
        return value, x_star
    
        
    def greedy(self, cost_matrix, n):
        """
        first stage : greedy initialization
        cost_matrix : np.ndarray
        n : selection budget
        """
        pbar = tqdm(total=n)
        flag = True
        T_len, V_len = cost_matrix.shape
        S = []      # subset initialization
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        cost_gpu = torch.from_numpy(cost_matrix).to(device)

        # First, we add the row with the smallest sum to S
        idx = cost_matrix.sum(axis=1).argmin().item()
        S.append(idx)
        pbar.update(1)
        cost_min = cost_gpu[idx]  # The column-wise minimum of the cost matrix, constrained to the rows in S
        cost_sum = cost_gpu.sum(dim=1)
        w = torch.zeros(T_len) 
        min_indices = cost_gpu.argmin(dim=0).detach().cpu() 
        unique_indices, counts = torch.unique(min_indices, return_counts=True)
        w[unique_indices] = counts.float()/V_len
        if (w>0).sum() <= n:
            flag = False

        with torch.no_grad():
            try:
                while True:
                    if len(S) >= n:
                        w = torch.zeros(T_len)
                        min_indices = cost_gpu[S, :].argmin(axis=0).detach().cpu()  
                        unique_indices, counts = torch.unique(min_indices, return_counts=True)  
                        w[torch.tensor(S)[unique_indices]] = counts.float() / V_len
                        if (w>0).sum() >= n or not flag:
                            w = w.cpu().numpy()
                            if flag:
                                S = [i for i in S if w[i]>0]
                            return S

                    # compute Gain and select the row idx with smallest Gain
                    # we ultilize an equivalent form of Gain in Eq. 14, which can further reduce repeated computation
                    # one can get this form by writing (x, 0)^- as (x-|x|)/2
                    diff_set =  torch.from_numpy(np.setdiff1d(np.arange(T_len), np.array(S)))
                    gain = (cost_sum - (cost_min-cost_gpu).abs().sum(dim=1)).cpu()[diff_set]
                    e = diff_set[gain.argmin()].item()

                    # Update S and the column-wise minimum of the cost matrix constrained to the rows in S
                    S.append(e)
                    cost_min = cost_gpu[S].min(dim=0)[0]
                    pbar.update(1)
            finally:   
                cost_gpu.cpu()  
                cost_min.cpu()   
                del cost_gpu, cost_min, 
                torch.cuda.empty_cache()   
              
       
    def compute_f_matrix(self, cost_matrix, x_star, S):
        """
        compute f matrix efficiently in parallel
        """
        cost_matrix.sub_(x_star.unsqueeze(1))   # to save cuda memory
        cost_min_S = cost_matrix[S, :].min(dim=0, keepdim=True)[0]
        mask = torch.ones_like(cost_matrix).bool()   
        row_in_S_equal_S_min = (cost_matrix[S, :] == cost_min_S)
        num_row_in_S_equal_S_min = row_in_S_equal_S_min.sum(dim=0)
        cost_second_min_S = torch.where(         # compute the column-wise second minimum of the cost matrix constrained to the rows in S 
            row_in_S_equal_S_min,  
            torch.finfo(cost_matrix[S,:].dtype).max, 
            cost_matrix[S,:]  
        ).min(dim=0, keepdim=True)[0] 
        mask[S, :][:, num_row_in_S_equal_S_min==1][row_in_S_equal_S_min[:, num_row_in_S_equal_S_min==1]] = False 
        cost_matrix.add_(x_star.unsqueeze(1))
        return torch.where(mask, cost_min_S,  cost_second_min_S)
    
    def compute_F(self, cost_matrix, f_matrix, y_hat, S_len):
        """
        compute F in Eq.16
        """
        cost_matrix.sub_(y_hat.unsqueeze(1))
        F_scores = torch.minimum(cost_matrix-f_matrix, torch.tensor(0)).mean(dim=1) + y_hat/S_len
        cost_matrix.add_(y_hat.unsqueeze(1))
        return F_scores
        
        
                
    def pruning(self, cost_matrix, S, x_star, cut_num=30):
        """
        estimate MI score by Theorem 4.3 and then conduct inner pruning & outer pruning
        
        S : current coreset
        x_star : optimal dual variables associated with mu_S
        cut_num : the number of exchange candidates budget
        """
        T_len, V_len = cost_matrix.shape
        S_len = len(S)
        R = math.ceil(V_len/S_len)
        
        # compute f matrix in parallel
        f_matrix = self.compute_f_matrix(cost_matrix, x_star, S)   
        # compute y_hat, and we only need to find the top-R values for each row
        y_hat = (cost_matrix - f_matrix).topk(k=R, dim=1, largest=True)[0][:, -1]
        # estimate MI by Eq.16
        estimate_MI_scores = self.compute_F(cost_matrix, f_matrix, y_hat, S_len)
        
        # outer pruning
        diff = np.setdiff1d(np.arange(T_len), np.array(S))
        outer_scores = estimate_MI_scores[diff].cpu()  
        outer_idx = diff[outer_scores.argsort()[:cut_num]] 
        
        # inner pruning
        S = np.array(S)
        inner_scores = estimate_MI_scores[S].cpu()
        inner_idx = S[inner_scores.argsort()[-cut_num:]]

        return list(inner_idx), list(outer_idx)
    
    
    def refine(self, cost_matrix, S, max_iters=100, cut_num=30, plot_descent_curve=False, print_pass_1=False):
        """
        stage 2: refinement via sample exchange accelerated by pruning strategies
        """
        pass_1_num = 0
        s_list = []
        pbar = tqdm(total=max_iters)
        iters = 0
        
        s, x_star = self.compute_ot(cost_matrix, S)
        s_list.append(s)
        while True:
            
            if iters >= max_iters:
                print(f"------ final score : {s} ------")
                break
                
            print(f"------ score of iter {iters} : {s} ------")
            pass_1_flag = True
            early_stop = True
            
            inner_idx, outer_idx = self.pruning(torch.from_numpy(cost_matrix).cuda(), S, torch.from_numpy(x_star).cuda(), cut_num=cut_num)
            
            for (i, o) in list(itertools.product(inner_idx, outer_idx)):
                    S_ = S.copy()
                    S_.remove(i)
                    S_.append(o)
                    results = self.compute_ot(cost_matrix, S_)
                    if results[0] < s:
                        s, x_star = results
                        S.remove(i)
                        S.append(o)
                        iters += 1
                        early_stop = False
                        pbar.update(1) 
                        s_list.append(s)
                        break
                    pass_1_flag = False
            if pass_1_flag:
                pass_1_num += 1
            if early_stop:
                break
        
        if plot_descent_curve:
            plt.plot(s_list)
        if len(s_list) > 0 and print_pass_1:
            print(f"------ pass@1 : {pass_1_num/(len(s_list)-1)} ------")
        return S
    
    
    def coreset_select(self, 
                       lamda=0, 
                       n=1024,
                       max_iters=100, 
                       cut_num=30, 
                       save_subset=False,
                       plot_descent_curve=False,
                       print_pass_1=False,
                       dist_matrix=None,
                       grads=None
                      ):
        """
        Algorithm 1.
        
        lamda : hyper-parameter in POO
        n : selection budget
        max_iters : max number of exchange iterations
        cut_num : the number of remained samples after inner/outer pruning
        """
        self.lamda = lamda
        self.n = n
        self.max_iters = max_iters
        self.cut_num = cut_num
        
        if dist_matrix is None:
            dist_matrix = self.dist_matrix
        if grads is None:
            grads = self.grads
        
        cost_matrix = (dist_matrix - self.lamda*grads[:, None]).numpy()
        cost_matrix = (cost_matrix - cost_matrix.min())/(cost_matrix.max()-cost_matrix.min())
        
        S_gre = self.greedy(cost_matrix, self.n)  # stage 1: greedy initialization
        print(f"------ score of S_gre : {self.compute_ot(cost_matrix, S_gre)[0]} ------")
        S_opt = self.refine(cost_matrix, S_gre, max_iters=self.max_iters, cut_num=self.cut_num, plot_descent_curve=plot_descent_curve, print_pass_1=print_pass_1)
        
        if save_subset and self.save_dir_path is not None and self.train_data_path is not None:
            os.makedirs(self.save_dir_path, exist_ok=True)
            with open(self.train_data_path, "r") as f:
                train_data = json.load(f)
            subset_data = [train_data[i] for i in S_opt]
            with open(os.path.join(self.save_dir_path, f"n={n}, lamda={lamda}, max_iters={max_iters}.json"), "w") as f:
                json.dump(subset_data, f, indent=4, ensure_ascii=False)
            
        return S_opt
    
    
    
    
    
        
        
        
class LabelEnhancedITRA():
    
    def __init__(self, train_data_path=None, valid_data_path=None, save_dir_path=None, train_embedding_path=None, valid_embedding_path=None, grads_path=None):
        self.train_data_path = train_data_path
        self.valid_data_path = valid_data_path
        self.save_dir_path = save_dir_path
        self.train_emb_path = train_embedding_path
        self.valid_emb_path = valid_embedding_path
        self.grads_path = grads_path
        
        if train_data_path is not None:
            with open(train_data_path, "r", encoding="utf-8") as f:
                self.train_label = torch.tensor([int(data["output"]=="Yes") for data in json.load(f)])
        if valid_data_path is not None:
            with open(valid_data_path, "r") as f:
                self.valid_label = torch.tensor([int(data["output"]=="Yes") for data in json.load(f)])
        
        if train_embedding_path is not None:
            embeddings_train = torch.load(train_embedding_path)
        if valid_embedding_path is not None:
            embeddings_valid = torch.load(valid_embedding_path)                
        if train_embedding_path is not None and valid_embedding_path is not None:
            dist_matrix = torch.cdist(embeddings_train, embeddings_valid, p=2).cpu()
            if train_data_path is not None and valid_data_path is not None:
                self.dist_matrix_pos = dist_matrix[self.train_label==1, :][:, self.valid_label==1]
                self.dist_matrix_neg = dist_matrix[self.train_label==0, :][:, self.valid_label==0]
            else:
                self.dist_matrix = dist_matrix
        
        if grads_path is not None:
            grads = torch.load(grads_path)
            grads = grads/grads.max()
            if train_data_path is not None and valid_data_path is not None:
                self.grads_pos = grads[self.train_label==1]
                self.grads_neg = grads[self.train_label==0]
            else:
                self.grads = grads
         
        
        
        
    def coreset_select(self, 
                       lamda=0, 
                       n=64,
                       max_iters=100, 
                       cut_num=30, 
                       save_subset=False,
                      ):
        """
        Algorithm 2.
        We implemented a binary classification algorithm here, as this is the setup for the CTRPre experiment in our paper. 
        The code for K-class classification can be easily implemented following this binary version, and we will release the 
        K-classification version in the GitHub repository after the paper is accepted.
        
        lamda : hyper-parameter in POO
        n : selection budget
        max_iters : max number of exchange iterations
        cut_n
        """
        self.lamda = lamda
        self.n = n
        self.max_iters = max_iters
        self.cut_num = cut_num
        
        self.n_pos = round(self.valid_label.float().mean().item()*self.n)
        self.n_neg = self.n - self.n_pos
        
        ITRA_solver = ITRA()
        
        # pos
        print("------ start conducting coreset selection within class = pos ------")
        print(f"------ the selection budget is {self.n_pos} ------")
        pos_idx_dataset = np.arange(len(self.train_label))[self.train_label==1]
        S_pos = ITRA_solver.coreset_select(lamda=self.lamda, n=self.n_pos, max_iters=self.max_iters, cut_num=self.cut_num,
                                             save_subset=False, plot_descent_curve=False, 
                                              dist_matrix=self.dist_matrix_pos, grads=self.grads_pos)
        S_pos = pos_idx_dataset[S_pos].tolist()
        
        
        # neg
        print("------ start conducting coreset selection within class = neg ------")
        print(f"------ the selection budget is {self.n_neg} ------")
        neg_idx_dataset = np.arange(len(self.train_label))[self.train_label==0]
        S_neg = ITRA_solver.coreset_select(lamda=self.lamda, n=self.n_neg, max_iters=self.max_iters, cut_num=self.cut_num,
                                             save_subset=False, plot_descent_curve=False, 
                                              dist_matrix=self.dist_matrix_neg, grads=self.grads_neg)
        S_neg = neg_idx_dataset[S_neg].tolist()
        
        S_opt = S_pos + S_neg
        
        if save_subset and self.save_dir_path is not None and self.train_data_path is not None:
            os.makedirs(self.save_dir_path, exist_ok=True)
            with open(self.train_data_path, "r") as f:
                train_data = json.load(f)
            subset_data = [train_data[i] for i in S_opt]
            with open(os.path.join(self.save_dir_path, f"n={n}, lamda={lamda}, max_iters={max_iters}.json"), "w") as f:
                json.dump(subset_data, f, indent=4, ensure_ascii=False)
                
        return S_opt
        