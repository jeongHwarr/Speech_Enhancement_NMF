"""
NMF module.

@author: Jeonghwa Yoo
"""

import numpy as np
import sys
# from distfit import distfit
import scipy.stats as stats   


def nmf_train(V, max_iter, epsilon, num_basis):
    W = np.random.rand(V.shape[0], num_basis)+sys.float_info.epsilon
    H = np.random.rand(num_basis, V.shape[1])+sys.float_info.epsilon
       
    for i in range(max_iter):
        Wt = np.transpose(W)
        Ht = np.transpose(H)
        
        W = W * np.matmul(V, Ht)/ np.matmul(W, np.matmul(H, Ht))
        H = H * np.matmul(Wt, V) / np.matmul(np.matmul(Wt, W), H)
        
        V_hat = np.matmul(W,H)
        cost = get_loss(V,V_hat) # KL divergence
        print("Iteration: %03d, cost: %s" %(i+1, cost))

        if cost < epsilon:
            break
    return W,H

    
def nmf_test(V, W_train, H_train, max_iter, epsilon, penalty, algorithm='NMF'):       
    # initial basis vector(W) and encoding vector(H)
    W = W_train
    H = np.random.rand(W.shape[1],V.shape[1])+sys.float_info.epsilon #size: (basis,time)
    
    Wt = np.transpose(W)
    ONES = np.ones(V.shape)
 
    #-----------------# standard NMF (with multiplicative update rules)----------------- 
    if algorithm=='NMF': 
        for i in range(max_iter):
            
            H = H * np.matmul(Wt, V) / np.matmul(np.matmul(Wt, W), H)          
            V_hat = np.matmul(W,H)
            
            cost = get_loss(V,V_hat) # KL divergence
            # print("[Test] Iteration: %03d, cost: %s" %(i+1, cost))
            if cost < epsilon:
                break
        return H 
    
    #-----------------# NMF with gamma distribution----------------- 
    elif algorithm=='NMF_g':
        fit_alphas = np.zeros(H.shape)
        fit_locs = np.zeros(H.shape)
        fit_betas = np.zeros(H.shape)
        
        for i in range (H_train.shape[0]):
            fit_alpha, fit_loc, fit_beta=stats.gamma.fit(H_train[i,:])
            fit_alphas[i,:] = fit_alpha # shape parameter 
            fit_locs[i,:] = fit_loc
            fit_betas[i,:] = fit_beta # scale parameter
        
        for i in range(max_iter):
            penalty_term_1 = penalty*((1-fit_alphas)/H)
            penalty_term_2 = penalty*(1/fit_betas);
            penalty_term = penalty_term_1 + penalty_term_2
            
            top = H*(np.matmul(Wt, V/(np.matmul(W,H)+sys.float_info.epsilon)))
            bottom = np.matmul(Wt,ONES)+penalty_term 
            H = top/bottom 
            
            V_hat = np.matmul(W,H)
            
            cost = get_loss(V,V_hat) # KL divergence
            # print("[Test] Iteration: %03d, cost: %s" %(i+1, cost))
            if cost < epsilon:
                break
        return H
            
    #-----------------# NMF with exponential distribution----------------- 
    elif algorithm=='NMF_e':
        param = np.zeros(H.shape)

        for i in range (H_train.shape[0]):
            param[i,:] = 1/(np.sum(H_train[i,:])/H_train.shape[1])
        
        for i in range(max_iter):
            penalty_term = penalty*(param);        
            
            top = H*(np.matmul(Wt, V/(np.matmul(W,H)+sys.float_info.epsilon)))
            bottom = np.matmul(Wt,ONES)+penalty_term 
            H = top/bottom 
                    
            V_hat = np.matmul(W,H)
            
            cost = get_loss(V,V_hat) # KL divergence
            # print("[Test] Iteration: %03d, cost: %s" %(i+1, cost))
            if cost < epsilon:
                break
        return H        
        
        

def get_loss(V,V_hat):
    loss = np.linalg.norm(V*np.log(V/V_hat)-V+V_hat,'fro')
    return loss    
        
        

    
    