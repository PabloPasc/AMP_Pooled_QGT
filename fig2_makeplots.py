#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 17:01:21 2023

@author: pp423
"""

from pool_amp import se_pool, amp_pool, create_B, X_iid_to_X_tilde, Y_iid_to_Y_iid_tilde, run_LP, run_NP, IHT_greedy
import numpy as np
import tikzplotlib
import matplotlib.pyplot as plt

#-------------------Fig. 2a - sigma=0------------------------------------------
#Simulations
alpha = 0.5
run_no = 100
sigma = 0
p = 500
max_iter = 200
num_mc_samples = 1000000
delta_array = np.sort(np.concatenate([[0.425, 0.45],np.linspace(0.1, 1, num=10)]))
delta_se_array = np.linspace(0.1, 1, 91)

#Green plot
pi = np.array([1/3, 1/3, 1/3])
L = len(pi)

delta_se_corr_amp = []
delta_corr_amp = []
delta_corr_std_amp = []
delta_corr_lp = []
delta_corr_std_lp = []
delta_corr_iht = []
delta_corr_std_iht = []

for delta in delta_se_array:
    print("delta: ", delta)
    n = int(delta*p)
    _, _, mse_est_array, corr_array = se_pool(delta, L, alpha, pi, sigma, max_iter, num_mc_samples)
    delta_se_corr_amp.append(corr_array[-1])

for delta in delta_array:
    print("delta: ", delta)
    n = int(delta*p)
    
    delta_corr_runs_amp = []
    delta_corr_runs_lp = []
    delta_corr_runs_iht = []
    
    for run in range(run_no):
        
        B_0 = create_B(pi, p)
        
        print("Run: ", run)
        #AMP - on Bernoulli matrix
        X = np.random.binomial(1, alpha, (n,p))
        Psi = np.random.normal(0, np.sqrt(p)*sigma, (n,L))
        Y = np.dot(X, B_0) + Psi
        sparsity_lvls = np.einsum('i,ij->j', np.ones(p), B_0)
        pi_true = (1/p)*sparsity_lvls
        
        X_tilde = X_iid_to_X_tilde(X, alpha)
        Y_tilde = Y_iid_to_Y_iid_tilde(Y, alpha, n, p, pi_true)
        B, _, mse_arr, noise_arr, mse_final_b = amp_pool(pi, X_tilde, Y_tilde, max_iter, B_0)
        corr_av_b = np.mean(np.einsum('ij, ij->i', B, B_0))
        print(mse_final_b)
        delta_corr_runs_amp.append(corr_av_b)
        
        #LP - with pi
        B_LP_est = run_LP(n, p, L, Y, X, pi)
        corr_av_lp = np.mean(np.einsum('ij, ij->i', B_LP_est, B_0))
        delta_corr_runs_lp.append(corr_av_lp)
        
        # IHT
        sparsity_lvls = (pi_true*p).round().astype(int)
        B_hat = IHT_greedy(Y_tilde, X_tilde, p, sparsity_lvls, num_iter=100)
        corr_av_iht = np.mean(np.einsum('ij, ij->i', B_hat, B_0))
        delta_corr_runs_iht.append(corr_av_iht)
        
    
    delta_corr_amp.append(np.mean(delta_corr_runs_amp))
    delta_corr_std_amp.append(np.std(delta_corr_runs_amp))
    delta_corr_lp.append(np.mean(delta_corr_runs_lp))
    delta_corr_std_lp.append(np.std(delta_corr_runs_lp))
    delta_corr_iht.append(np.mean(delta_corr_runs_iht))
    delta_corr_std_iht.append(np.std(delta_corr_runs_iht))
    
plt.figure()
plt.plot(delta_se_array, delta_se_corr_amp, label='SE', color = 'blue', linestyle = 'dashed')
plt.errorbar(delta_array, delta_corr_amp, yerr=delta_corr_std_amp, label =r"AMP", fmt='*', color='blue',ecolor='lightblue', elinewidth=3, capsize=0)
plt.errorbar(delta_array, delta_corr_lp, yerr=delta_corr_std_lp, label =r"LP", fmt='*', color='red',ecolor='coral', elinewidth=3, capsize=0, linestyle='dotted')
plt.errorbar(delta_array, delta_corr_iht, yerr=delta_corr_std_iht, label =r"IHT", fmt='*', color='green',ecolor='lightgreen', elinewidth=3, capsize=0, linestyle='dotted')
plt.grid(alpha=0.4)
plt.legend()
plt.ylabel('Correlation')
plt.xlabel(r'$\delta=n/p$')
#tikzplotlib.save("pool_fig2a.tex")


#-------------------Fig. 2b - sigma=0.1------------------------------------------
#Simulations
alpha = 0.5
run_no = 100
sigma = 0.1
p = 500
max_iter = 200
num_mc_samples = 1000000
delta_array = np.sort(np.concatenate([[0.65],np.linspace(0.1, 1, num=10)]))
delta_se_array = np.linspace(0.1, 1, 91)

#Green plot
pi = np.array([1/3, 1/3, 1/3])
L = len(pi)

delta_se_corr_amp = []
delta_corr_amp = []
delta_corr_std_amp = []
delta_corr_amp_nopi = []
delta_corr_std_amp_nopi = []
delta_corr_np = []
delta_corr_std_np = []
delta_corr_iht = []
delta_corr_std_iht = []

for delta in delta_se_array:
    print("delta: ", delta)
    n = int(delta*p)
    _, _, mse_est_array, corr_array = se_pool(delta, L, alpha, pi, sigma, max_iter, num_mc_samples)
    delta_se_corr_amp.append(corr_array[-1])

for delta in delta_array:
    print("delta: ", delta)
    n = int(delta*p)
    
    delta_corr_runs_amp = []
    delta_corr_runs_amp_nopi = []
    delta_corr_runs_np = []
    delta_corr_runs_iht = []
    
    for run in range(run_no):
        
        B_0 = create_B(pi, p)
        
        print("Run: ", run)
        #AMP - on Bernoulli matrix
        X = np.random.binomial(1, alpha, (n,p))
        Psi = np.random.normal(0, np.sqrt(p)*sigma, (n,L))
        Y = np.dot(X, B_0) + Psi
        sparsity_lvls = np.einsum('i,ij->j', np.ones(p), B_0)
        pi_true = (1/p)*sparsity_lvls
        
        X_tilde = X_iid_to_X_tilde(X, alpha)
        Y_tilde = Y_iid_to_Y_iid_tilde(Y, alpha, n, p, pi_true)
        B, _, mse_arr, noise_arr, mse_final_b = amp_pool(pi, X_tilde, Y_tilde, max_iter, B_0)
        corr_av_b = np.mean(np.einsum('ij, ij->i', B, B_0))
        print(mse_final_b)
        delta_corr_runs_amp.append(corr_av_b)
        
        #NP - with pi
        B_NP_est = run_NP(n, p, L, Y, X, sigma, pi)
        corr_av_np = np.mean(np.einsum('ij, ij->i', B_NP_est, B_0))
        delta_corr_runs_np.append(corr_av_np)
        
        # IHT
        sparsity_lvls = (pi_true*p).round().astype(int)
        B_hat = IHT_greedy(Y_tilde, X_tilde, p, sparsity_lvls, num_iter=100)
        corr_av_iht = np.mean(np.einsum('ij, ij->i', B_hat, B_0))
        delta_corr_runs_iht.append(corr_av_iht)
        
        #AMP - on Bernoulli matrix without pi_true
        Y_tilde_pi = Y_iid_to_Y_iid_tilde(Y, alpha, n, p, pi)
        B, _, mse_arr, noise_arr, mse_final_b = amp_pool(pi, X_tilde, Y_tilde_pi, max_iter, B_0)
        corr_av_b_nopi = np.mean(np.einsum('ij, ij->i', B, B_0))
        print(mse_final_b)
        delta_corr_runs_amp_nopi.append(corr_av_b_nopi)
        
    
    delta_corr_amp.append(np.mean(delta_corr_runs_amp))
    delta_corr_std_amp.append(np.std(delta_corr_runs_amp))
    delta_corr_amp_nopi.append(np.mean(delta_corr_runs_amp_nopi))
    delta_corr_std_amp_nopi.append(np.std(delta_corr_runs_amp_nopi))
    delta_corr_np.append(np.mean(delta_corr_runs_np))
    delta_corr_std_np.append(np.std(delta_corr_runs_np))
    delta_corr_iht.append(np.mean(delta_corr_runs_iht))
    delta_corr_std_iht.append(np.std(delta_corr_runs_iht))
    
    
plt.figure()
plt.plot(delta_se_array, delta_se_corr_amp, label='SE', color = 'blue', linestyle = 'dashed')
plt.errorbar(delta_array, delta_corr_amp, yerr=delta_corr_std_amp, label =r"AMP, $\hat{\pi}$", fmt='*', color='blue',ecolor='lightblue', elinewidth=3, capsize=0)
#plt.errorbar(delta_array, delta_corr_amp_nopi, yerr=delta_corr_std_amp_nopi, label =r"AMP, $\pi$", fmt='*', color='black',ecolor='lightgrey', elinewidth=3, capsize=0)
plt.errorbar(delta_array, delta_corr_np, yerr=delta_corr_std_np, label =r"CVX", fmt='*', color='red',ecolor='coral', elinewidth=3, capsize=0, linestyle='dotted')
plt.errorbar(delta_array, delta_corr_iht, yerr=delta_corr_std_iht, label =r"IHT", fmt='*', color='green',ecolor='lightgreen', elinewidth=3, capsize=0, linestyle='dotted')
plt.grid(alpha=0.4)
plt.legend()
plt.ylabel('Correlation')
plt.xlabel(r'$\delta=n/p$')
#tikzplotlib.save("pool_fig2b.tex")

#-------------------Fig. 2c - sigma=0.3------------------------------------------
#Simulations
alpha = 0.5
run_no = 100
sigma = 0.3
p = 500
max_iter = 200
num_mc_samples = 1000000
delta_array = np.linspace(0.5, 8, num=16)
delta_se_array = np.linspace(0.1, 8, 80)

#Green plot
pi = np.array([1/3, 1/3, 1/3])
L = len(pi)

delta_se_corr_amp = []
delta_corr_amp = []
delta_corr_std_amp = []
delta_corr_amp_nopi = []
delta_corr_std_amp_nopi = []
delta_corr_np = []
delta_corr_std_np = []
delta_corr_iht = []
delta_corr_std_iht = []

for delta in delta_se_array:
    print("delta: ", delta)
    n = int(delta*p)
    _, _, mse_est_array, corr_array = se_pool(delta, L, alpha, pi, sigma, max_iter, num_mc_samples)
    delta_se_corr_amp.append(corr_array[-1])

for delta in delta_array:
    print("delta: ", delta)
    n = int(delta*p)
    
    delta_corr_runs_amp = []
    delta_corr_runs_amp_nopi = []
    delta_corr_runs_np = []
    delta_corr_runs_iht = []
    
    for run in range(run_no):
        
        B_0 = create_B(pi, p)
        
        print("Run: ", run)
        #AMP - on Bernoulli matrix
        X = np.random.binomial(1, alpha, (n,p))
        Psi = np.random.normal(0, np.sqrt(p)*sigma, (n,L))
        Y = np.dot(X, B_0) + Psi
        sparsity_lvls = np.einsum('i,ij->j', np.ones(p), B_0)
        pi_true = (1/p)*sparsity_lvls
        
        X_tilde = X_iid_to_X_tilde(X, alpha)
        Y_tilde = Y_iid_to_Y_iid_tilde(Y, alpha, n, p, pi_true)
        B, _, mse_arr, noise_arr, mse_final_b = amp_pool(pi, X_tilde, Y_tilde, max_iter, B_0)
        corr_av_b = np.mean(np.einsum('ij, ij->i', B, B_0))
        print(mse_final_b)
        delta_corr_runs_amp.append(corr_av_b)
        
        #NP - with pi
        B_NP_est = run_NP(n, p, L, Y, X, sigma, pi)
        corr_av_np = np.mean(np.einsum('ij, ij->i', B_NP_est, B_0))
        delta_corr_runs_np.append(corr_av_np)
        
        # IHT
        sparsity_lvls = (pi_true*p).round().astype(int)
        B_hat = IHT_greedy(Y_tilde, X_tilde, p, sparsity_lvls, num_iter=100)
        corr_av_iht = np.mean(np.einsum('ij, ij->i', B_hat, B_0))
        delta_corr_runs_iht.append(corr_av_iht)
        
        #AMP - on Bernoulli matrix without pi_true
        Y_tilde_pi = Y_iid_to_Y_iid_tilde(Y, alpha, n, p, pi)
        B, _, mse_arr, noise_arr, mse_final_b = amp_pool(pi, X_tilde, Y_tilde_pi, max_iter, B_0)
        corr_av_b_nopi = np.mean(np.einsum('ij, ij->i', B, B_0))
        print(mse_final_b)
        #delta_mse_b.append(mse_final_b)
        delta_corr_runs_amp_nopi.append(corr_av_b_nopi)
        
    
    delta_corr_amp.append(np.mean(delta_corr_runs_amp))
    delta_corr_std_amp.append(np.std(delta_corr_runs_amp))
    delta_corr_amp_nopi.append(np.mean(delta_corr_runs_amp_nopi))
    delta_corr_std_amp_nopi.append(np.std(delta_corr_runs_amp_nopi))
    delta_corr_np.append(np.mean(delta_corr_runs_np))
    delta_corr_std_np.append(np.std(delta_corr_runs_np))
    delta_corr_iht.append(np.mean(delta_corr_runs_iht))
    delta_corr_std_iht.append(np.std(delta_corr_runs_iht))
    
    
plt.figure()
plt.plot(delta_se_array, delta_se_corr_amp, label='SE', color = 'blue', linestyle = 'dashed')
plt.errorbar(delta_array, delta_corr_amp, yerr=delta_corr_std_amp, label =r"AMP, $\hat{\pi}$", fmt='*', color='blue',ecolor='lightblue', elinewidth=3, capsize=0)
#plt.errorbar(delta_array, delta_corr_amp_nopi, yerr=delta_corr_std_amp_nopi, label =r"AMP, $\pi$", fmt='*', color='black',ecolor='lightgrey', elinewidth=3, capsize=0)
plt.errorbar(delta_array, delta_corr_np, yerr=delta_corr_std_np, label =r"CVX", fmt='*', color='red',ecolor='coral', elinewidth=3, capsize=0, linestyle='dotted')
plt.errorbar(delta_array, delta_corr_iht, yerr=delta_corr_std_iht, label =r"IHT", fmt='*', color='green',ecolor='lightgreen', elinewidth=3, capsize=0, linestyle='dotted')
plt.grid(alpha=0.4)
plt.legend()
plt.ylabel('Correlation')
plt.xlabel(r'$\delta=n/p$')
#tikzplotlib.save("pool_fig2c.tex")