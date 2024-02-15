#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 12:14:15 2023

@author: pp423
"""

from pool_amp import se_pool, amp_pool, create_B, X_iid_to_X_tilde, Y_iid_to_Y_iid_tilde
import numpy as np
import tikzplotlib
import matplotlib.pyplot as plt

#-----------------------Fig. 3a-----------------------------------------------
#Simulations
alpha = 0.5
run_no = 100
p = 500
max_iter = 200
num_mc_samples = 1000000
delta_array = np.linspace(0.1, 1, num=10)
delta_se_array = np.linspace(0.1, 1, 91)

pi = np.array([0.5, 0.5])
L = len(pi)

#Green plot
sigma = 0.5
delta_se_corr_green = []
delta_corr_green = []
delta_corr_std_green = []


for delta in delta_se_array:
    print("delta: ", delta)
    n = int(delta*p)
    _, _, mse_est_array, corr_array = se_pool(delta, L, alpha, pi, sigma, max_iter, num_mc_samples)
    delta_se_corr_green.append(corr_array[-1])

for delta in delta_array:
    print("delta: ", delta)
    n = int(delta*p)
    
    delta_corr_runs_green = []
    
    for run in range(run_no):
        
        B_0 = create_B(pi, p)
        
        print("Run: ", run)
        #AMP - on Bernoulli matrix
        X = np.random.binomial(1, alpha, (n,p))
        Psi = np.random.normal(0, np.sqrt(p)*sigma, (n,L))
        Y = np.dot(X, B_0) + Psi
        pi_true = (1/p)*np.einsum('i,ij->j', np.ones(p), B_0)
        
        X_tilde = X_iid_to_X_tilde(X, alpha)
        Y_tilde = Y_iid_to_Y_iid_tilde(Y, alpha, n, p, pi_true)
        B, _, mse_arr, noise_arr, mse_final_b = amp_pool(pi, X_tilde, Y_tilde, max_iter, B_0)
        corr_av_b = np.mean(np.einsum('ij, ij->i', B, B_0))
        print(mse_final_b)
        delta_corr_runs_green.append(corr_av_b)
        
        
        
    
    delta_corr_green.append(np.mean(delta_corr_runs_green))
    delta_corr_std_green.append(np.std(delta_corr_runs_green))
    
#Red plot
sigma = 0.3

delta_se_corr_red = []
delta_corr_red = []
delta_corr_std_red = []

for delta in delta_se_array:
    print("delta: ", delta)
    n = int(delta*p)
    _, _, mse_est_array, corr_array = se_pool(delta, L, alpha, pi, sigma, max_iter, num_mc_samples)
    delta_se_corr_red.append(corr_array[-1])

for delta in delta_array:
    print("delta: ", delta)
    n = int(delta*p)
    
    delta_corr_runs_red = []
    
    for run in range(run_no):
        
        B_0 = create_B(pi, p)
        
        print("Run: ", run)
        #AMP - on Bernoulli matrix
        X = np.random.binomial(1, alpha, (n,p))
        Psi = np.random.normal(0, np.sqrt(p)*sigma, (n,L))
        Y = np.dot(X, B_0) + Psi
        pi_true = (1/p)*np.einsum('i,ij->j', np.ones(p), B_0)
        
        X_tilde = X_iid_to_X_tilde(X, alpha)
        Y_tilde = Y_iid_to_Y_iid_tilde(Y, alpha, n, p, pi_true)
        B, _, mse_arr, noise_arr, mse_final_b = amp_pool(pi, X_tilde, Y_tilde, max_iter, B_0)
        corr_av_b = np.mean(np.einsum('ij, ij->i', B, B_0))
        print(mse_final_b)
        delta_corr_runs_red.append(corr_av_b)
        
        
        
    
    delta_corr_red.append(np.mean(delta_corr_runs_red))
    delta_corr_std_red.append(np.std(delta_corr_runs_red))
    

#Blue plot
sigma = 0.1

delta_se_corr_blue = []
delta_corr_blue = []
delta_corr_std_blue = []

for delta in delta_se_array:
    print("delta: ", delta)
    n = int(delta*p)
    _, _, mse_est_array, corr_array = se_pool(delta, L, alpha, pi, sigma, max_iter, num_mc_samples)
    delta_se_corr_blue.append(corr_array[-1])

for delta in delta_array:
    print("delta: ", delta)
    n = int(delta*p)
    
    delta_corr_runs_blue = []
    
    for run in range(run_no):
        
        B_0 = create_B(pi, p)
        
        print("Run: ", run)
        #AMP - on Bernoulli matrix
        X = np.random.binomial(1, alpha, (n,p))
        Psi = np.random.normal(0, np.sqrt(p)*sigma, (n,L))
        Y = np.dot(X, B_0) + Psi
        pi_true = (1/p)*np.einsum('i,ij->j', np.ones(p), B_0)
        
        X_tilde = X_iid_to_X_tilde(X, alpha)
        Y_tilde = Y_iid_to_Y_iid_tilde(Y, alpha, n, p, pi_true)
        B, _, mse_arr, noise_arr, mse_final_b = amp_pool(pi, X_tilde, Y_tilde, max_iter, B_0)
        corr_av_b = np.mean(np.einsum('ij, ij->i', B, B_0))
        print(mse_final_b)
        delta_corr_runs_blue.append(corr_av_b)
        
    
    delta_corr_blue.append(np.mean(delta_corr_runs_blue))
    delta_corr_std_blue.append(np.std(delta_corr_runs_blue))
    
     
plt.figure()
plt.plot(delta_se_array, delta_se_corr_blue, label=r'SE, $\sigma$=0.1', color = 'blue', linestyle = 'dashed')
plt.plot(delta_se_array, delta_se_corr_red, label=r'SE, $\sigma$=0.3', color = 'red', linestyle = 'dashed')
plt.plot(delta_se_array, delta_se_corr_green, label=r'SE, $\sigma$=0.5', color = 'green', linestyle = 'dashed')
plt.errorbar(delta_array, delta_corr_blue, yerr=delta_corr_std_blue, label =r"AMP, $\sigma$=0.1", fmt='*', color='blue',ecolor='lightblue', elinewidth=3, capsize=0)
plt.errorbar(delta_array, delta_corr_red, yerr=delta_corr_std_red, label =r"AMP, $\sigma$=0.3", fmt='*', color='red',ecolor='coral', elinewidth=3, capsize=0)
plt.errorbar(delta_array, delta_corr_green, yerr=delta_corr_std_green, label =r"AMP, $\sigma$=0.5", fmt='*', color='green',ecolor='lightgreen', elinewidth=3, capsize=0)
plt.grid(alpha=0.4)
plt.legend()
plt.ylabel('Correlation')
plt.xlabel(r'$\delta=n/p$')
#tikzplotlib.save("pool_fig3a.tex")


#-----------------------------------Fig.3b------------------------------------- 

pi = np.array([1/3, 1/3, 1/3])
L = len(pi)

#Green plot
sigma = 0.5

delta_se_corr_greenb = []
delta_corr_greenb = []
delta_corr_std_greenb = []

for delta in delta_se_array:
    print("delta: ", delta)
    n = int(delta*p)
    _, _, mse_est_array, corr_array = se_pool(delta, L, alpha, pi, sigma, max_iter, num_mc_samples)
    delta_se_corr_greenb.append(corr_array[-1])

for delta in delta_array:
    print("delta: ", delta)
    n = int(delta*p)
    
    delta_corr_runs_greenb = []
    
    for run in range(run_no):
        
        B_0 = create_B(pi, p)
        
        print("Run: ", run)
        #AMP - on Bernoulli matrix
        X = np.random.binomial(1, alpha, (n,p))
        Psi = np.random.normal(0, np.sqrt(p)*sigma, (n,L))
        Y = np.dot(X, B_0) + Psi
        pi_true = (1/p)*np.einsum('i,ij->j', np.ones(p), B_0)
        
        X_tilde = X_iid_to_X_tilde(X, alpha)
        Y_tilde = Y_iid_to_Y_iid_tilde(Y, alpha, n, p, pi_true)
        B, _, mse_arr, noise_arr, mse_final_b = amp_pool(pi, X_tilde, Y_tilde, max_iter, B_0)
        corr_av_b = np.mean(np.einsum('ij, ij->i', B, B_0))
        print(mse_final_b)
        delta_corr_runs_greenb.append(corr_av_b)
        
    
    delta_corr_greenb.append(np.mean(delta_corr_runs_greenb))
    delta_corr_std_greenb.append(np.std(delta_corr_runs_greenb))
    


#Red plot
sigma = 0.3

delta_se_corr_redb = []
delta_corr_redb = []
delta_corr_std_redb = []

for delta in delta_se_array:
    print("delta: ", delta)
    n = int(delta*p)
    _, _, mse_est_array, corr_array = se_pool(delta, L, alpha, pi, sigma, max_iter, num_mc_samples)
    delta_se_corr_redb.append(corr_array[-1])

for delta in delta_array:
    print("delta: ", delta)
    n = int(delta*p)
    
    delta_corr_runs_redb = []
    
    for run in range(run_no):
        
        B_0 = create_B(pi, p)
        
        print("Run: ", run)
        #AMP - on Bernoulli matrix
        X = np.random.binomial(1, alpha, (n,p))
        Psi = np.random.normal(0, np.sqrt(p)*sigma, (n,L))
        Y = np.dot(X, B_0) + Psi
        pi_true = (1/p)*np.einsum('i,ij->j', np.ones(p), B_0)
        
        X_tilde = X_iid_to_X_tilde(X, alpha)
        Y_tilde = Y_iid_to_Y_iid_tilde(Y, alpha, n, p, pi_true)
        B, _, mse_arr, noise_arr, mse_final_b = amp_pool(pi, X_tilde, Y_tilde, max_iter, B_0)
        corr_av_b = np.mean(np.einsum('ij, ij->i', B, B_0))
        print(mse_final_b)
        delta_corr_runs_redb.append(corr_av_b)
        
        
    delta_corr_redb.append(np.mean(delta_corr_runs_redb))
    delta_corr_std_redb.append(np.std(delta_corr_runs_redb))
    

#Blue plot
sigma = 0.1

delta_se_corr_blueb = []
delta_corr_blueb = []
delta_corr_std_blueb = []

for delta in delta_se_array:
    print("delta: ", delta)
    n = int(delta*p)
    _, _, mse_est_array, corr_array = se_pool(delta, L, alpha, pi, sigma, max_iter, num_mc_samples)
    delta_se_corr_blueb.append(corr_array[-1])

for delta in delta_array:
    print("delta: ", delta)
    n = int(delta*p)
    
    delta_corr_runs_blueb = []
    
    for run in range(run_no):
        
        B_0 = create_B(pi, p)
        
        print("Run: ", run)
        #AMP - on Bernoulli matrix
        X = np.random.binomial(1, alpha, (n,p))
        Psi = np.random.normal(0, np.sqrt(p)*sigma, (n,L))
        Y = np.dot(X, B_0) + Psi
        pi_true = (1/p)*np.einsum('i,ij->j', np.ones(p), B_0)
        
        X_tilde = X_iid_to_X_tilde(X, alpha)
        Y_tilde = Y_iid_to_Y_iid_tilde(Y, alpha, n, p, pi_true)
        B, _, mse_arr, noise_arr, mse_final_b = amp_pool(pi, X_tilde, Y_tilde, max_iter, B_0)
        corr_av_b = np.mean(np.einsum('ij, ij->i', B, B_0))
        print(mse_final_b)
        delta_corr_runs_blueb.append(corr_av_b)
        
    
    delta_corr_blueb.append(np.mean(delta_corr_runs_blueb))
    delta_corr_std_blueb.append(np.std(delta_corr_runs_blueb))
    
     
plt.figure()
plt.plot(delta_se_array, delta_se_corr_blueb, label=r'SE, $\sigma$=0.1', color = 'blue', linestyle = 'dashed')
plt.plot(delta_se_array, delta_se_corr_redb, label=r'SE, $\sigma$=0.3', color = 'red', linestyle = 'dashed')
plt.plot(delta_se_array, delta_se_corr_greenb, label=r'SE, $\sigma$=0.5', color = 'green', linestyle = 'dashed')
plt.errorbar(delta_array, delta_corr_blueb, yerr=delta_corr_std_blueb, label =r"AMP, $\sigma$=0.1", fmt='*', color='blue',ecolor='lightblue', elinewidth=3, capsize=0)
plt.errorbar(delta_array, delta_corr_redb, yerr=delta_corr_std_redb, label =r"AMP, $\sigma$=0.3", fmt='*', color='red',ecolor='coral', elinewidth=3, capsize=0)
plt.errorbar(delta_array, delta_corr_greenb, yerr=delta_corr_std_greenb, label =r"AMP, $\sigma$=0.5", fmt='*', color='green',ecolor='lightgreen', elinewidth=3, capsize=0)
plt.grid(alpha=0.4)
plt.legend()
plt.ylabel('Correlation')
plt.xlabel(r'$\delta=n/p$')
#tikzplotlib.save("pool_fig3b.tex")



#---------------------------Fig. 4--------------------------------------------
#Simulations
alpha = 0.5
sigma = 0
run_no = 100
p = 500
max_iter = 200
num_mc_samples = 1000000
delta_array = np.linspace(0.1, 1, num=10)
delta_se_array = np.linspace(0.1, 1, 91)

pi = np.array([0.5, 0.5])
L = len(pi)

eps = 0.01

delta_corr_pihat = []
delta_corr_std_pihat = []
delta_corr_pihat_eps01 = []
delta_corr_std_pihat_eps01 = []
delta_corr_pihat_eps05 = []
delta_corr_std_pihat_eps05 = []

for delta in delta_array:
    print("delta: ", delta)
    n = int(delta*p)
    
    delta_corr_runs_pihat = []
    delta_corr_runs_pihat_eps01 = []
    delta_corr_runs_pihat_eps05 = []
    
    for run in range(run_no):
        
        B_0 = create_B(pi, p)
        
        print("Run: ", run)
        #AMP - on Bernoulli matrix
        X = np.random.binomial(1, alpha, (n,p))
        Psi = np.random.normal(0, np.sqrt(p)*sigma, (n,L))
        Y = np.dot(X, B_0) + Psi
        pi_true = (1/p)*np.einsum('i,ij->j', np.ones(p), B_0)
        
        pi_true_eps01 = pi_true + np.array([0.01, -0.01])
        pi_true_eps05 = pi_true + np.array([0.05, -0.05])
        
        X_tilde = X_iid_to_X_tilde(X, alpha)
        Y_tilde = Y_iid_to_Y_iid_tilde(Y, alpha, n, p, pi_true)
        B, _, mse_arr, noise_arr, mse_final_b = amp_pool(pi, X_tilde, Y_tilde, max_iter, B_0)
        corr_av = np.mean(np.einsum('ij, ij->i', B, B_0))
        print(mse_final_b)
        delta_corr_runs_pihat.append(corr_av)
        
        #eps = 0.01
        Y_tilde = Y_iid_to_Y_iid_tilde(Y, alpha, n, p, pi_true_eps01)
        B, _, mse_arr, noise_arr, mse_final_b = amp_pool(pi, X_tilde, Y_tilde, max_iter, B_0)
        corr_av = np.mean(np.einsum('ij, ij->i', B, B_0))
        print(mse_final_b)
        delta_corr_runs_pihat_eps01.append(corr_av)
        
        #eps = 0.05
        Y_tilde = Y_iid_to_Y_iid_tilde(Y, alpha, n, p, pi_true_eps05)
        B, _, mse_arr, noise_arr, mse_final_b = amp_pool(pi, X_tilde, Y_tilde, max_iter, B_0)
        corr_av = np.mean(np.einsum('ij, ij->i', B, B_0))
        print(mse_final_b)
        delta_corr_runs_pihat_eps05.append(corr_av)
        
    
    delta_corr_pihat.append(np.mean(delta_corr_runs_pihat))
    delta_corr_std_pihat.append(np.std(delta_corr_runs_pihat))
    delta_corr_pihat_eps01.append(np.mean(delta_corr_runs_pihat_eps01))
    delta_corr_std_pihat_eps01.append(np.std(delta_corr_runs_pihat_eps01))
    delta_corr_pihat_eps05.append(np.mean(delta_corr_runs_pihat_eps05))
    delta_corr_std_pihat_eps05.append(np.std(delta_corr_runs_pihat_eps05))


plt.figure()
plt.errorbar(delta_array, delta_corr_pihat, yerr=delta_corr_std_pihat, label =r"AMP, known $\hat{\pi}$", fmt='*', color='blue',ecolor='lightblue', elinewidth=3, capsize=0, linestyle='dashed')
plt.errorbar(delta_array, delta_corr_pihat_eps01, yerr=delta_corr_std_pihat_eps01, label =r"AMP, $\hat{\pi}\pm 0.01$", fmt='*', color='red',ecolor='coral', elinewidth=3, capsize=0, linestyle='dashed')
plt.errorbar(delta_array, delta_corr_pihat_eps05, yerr=delta_corr_std_pihat_eps05, label =r"AMP, $\hat{\pi}\pm 0.05$", fmt='*', color='green',ecolor='lightgreen', elinewidth=3, capsize=0, linestyle='dashed')
plt.grid(alpha=0.4)
plt.legend()
plt.ylabel('Correlation')
plt.xlabel(r'$\delta=n/p$')
#tikzplotlib.save("pool_fig4.tex")

