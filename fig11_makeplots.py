#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 11:34:39 2023

@author: pp423
"""

import numpy as np
from amp_qgt import g_in_bayes, Xiid_to_Xtilde, y_iid_to_y_iid_tilde, create_beta, fpr_fnr
from se_qgt import quantize
from gamp_qgt import gamp_unif_noise, state_ev_iid_gamp, run_BPDN
from numpy.random import binomial
import matplotlib.pyplot as plt

#-------------Fig. 11a - FPR vs FNR for GAMP vs BPDN, pi = 0.1----------------------------
thresh_array = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
alpha = 0.5
nu = 0.1
run_no = 100


lam_psi = 0.1
delta = 0.3
lim_const = lam_psi/np.sqrt(delta*alpha*(1-alpha))
p = 500
n = int(delta*p)

#State Evolution
fpr_se_01 = np.ones(len(thresh_array))
fnr_se_01 = np.ones(len(thresh_array))

n_mc = 100000
n_samples = 1000000
beta_0 = create_beta(nu, n_samples)
_, mse_pred, _, tau_array_se, tau =  state_ev_iid_gamp(delta, 200, nu, lim_const, n_mc)

print("MSE SE: ", mse_pred)
tau_G = np.sqrt(tau)*np.random.randn(n_samples)
beta_est = g_in_bayes(beta_0 + tau_G, tau, nu)
for i in range(len(thresh_array)):
    beta_q = quantize(beta_est, thresh_array[i])
    fpr_se_01[i], fnr_se_01[i], _ = fpr_fnr(beta_q, beta_0)
    

#AMP, BPDN
fpr_runs_01 = np.ones((run_no, len(thresh_array)))*5
fnr_runs_01 = np.ones((run_no, len(thresh_array)))*5
bpdn_fpr_runs_01 = np.ones((run_no, len(thresh_array)))*5
bpdn_fnr_runs_01 = np.ones((run_no, len(thresh_array)))*5

for run in range(run_no):  
    
    beta_0 = create_beta(nu, p) 
    
    X =  binomial(1, alpha, (n,p))
    psi = np.random.uniform(-lam_psi*np.sqrt(p), lam_psi*np.sqrt(p), n)
    y = np.dot(X, beta_0) + psi
    
    defect_no = np.sum(beta_0)
    
    X_tilde = Xiid_to_Xtilde(X, alpha)
    X_tilde_T = np.transpose(X_tilde)
    y_tilde = y_iid_to_y_iid_tilde(y, alpha, nu, n, p, defect_no)
    error_norm_array, nc_array, beta, tau_array = gamp_unif_noise(X_tilde, X_tilde_T, y_tilde, 200, nu, beta_0, lim_const)
    mse = (1/p)*(np.linalg.norm(beta - beta_0)**2)
    print("MSE GAMP: ", mse)
    
    
    #BPDN
    bound = lam_psi*np.sqrt(p)
    beta_bpdn = run_BPDN(n, p, X, y, bound)

    
    for i in range(len(thresh_array)):
        beta_amp_q = quantize(beta, thresh_array[i])
        fpr_runs_01[run, i], fnr_runs_01[run, i], _ = fpr_fnr(beta_amp_q, beta_0)
        beta_bpdn_q = quantize(beta_bpdn, thresh_array[i])
        bpdn_fpr_runs_01[run, i], bpdn_fnr_runs_01[run, i], _ = fpr_fnr(beta_bpdn_q, beta_0)

fpr_amp_01 = np.average(fpr_runs_01, axis = 0)
fnr_amp_01 = np.average(fnr_runs_01, axis = 0)
fpr_bpdn_01 = np.average(bpdn_fpr_runs_01, axis=0)
fnr_bpdn_01 = np.average(bpdn_fnr_runs_01, axis=0)

plt.figure()
plt.plot(fnr_se_01, fpr_se_01, marker='x', label=r'SE', linestyle='dashed', color='blue')
plt.plot(fnr_amp_01, fpr_amp_01, marker='o', label=r'GAMP', linestyle='none', color='blue')
plt.plot(fnr_bpdn_01, fpr_bpdn_01, marker='o', label=r'BPDN', linestyle='dotted', color='red')
plt.grid(alpha=0.4)
plt.legend()
plt.ylabel('FPR')
plt.xlabel('FNR')
#tikzplotlib.save("pool_fig11a.tex")

#---------------Fig. 11b - FPR vs FNR for GAMP vs BPDN, pi = 0.3---------------
nu = 0.3

#State Evolution
fpr_se_03 = np.ones(len(thresh_array))
fnr_se_03 = np.ones(len(thresh_array))

n_mc = 100000
n_samples = 1000000
beta_0 = create_beta(nu, n_samples)
_, mse_pred, _, tau_array_se, tau =  state_ev_iid_gamp(delta, 200, nu, lim_const, n_mc)

print("MSE SE: ", mse_pred)
tau_G = np.sqrt(tau)*np.random.randn(n_samples)
beta_est = g_in_bayes(beta_0 + tau_G, tau, nu)
for i in range(len(thresh_array)):
    beta_q = quantize(beta_est, thresh_array[i])
    fpr_se_03[i], fnr_se_03[i], _ = fpr_fnr(beta_q, beta_0)

#AMP, BPDN
fpr_runs_03 = np.ones((run_no, len(thresh_array)))*5
fnr_runs_03 = np.ones((run_no, len(thresh_array)))*5
bpdn_fpr_runs_03 = np.ones((run_no, len(thresh_array)))*5
bpdn_fnr_runs_03 = np.ones((run_no, len(thresh_array)))*5

for run in range(run_no):  
    
    beta_0 = create_beta(nu, p) 
    
    
    X =  binomial(1, alpha, (n,p))
    psi = np.random.uniform(-lam_psi*np.sqrt(p), lam_psi*np.sqrt(p), n)
    y = np.dot(X, beta_0) + psi
    
    defect_no = np.sum(beta_0)
    
    X_tilde = Xiid_to_Xtilde(X, alpha)
    X_tilde_T = np.transpose(X_tilde)
    y_tilde = y_iid_to_y_iid_tilde(y, alpha, nu, n, p, defect_no)
    error_norm_array, nc_array, beta, tau_array = gamp_unif_noise(X_tilde, X_tilde_T, y_tilde, 200, nu, beta_0, lim_const)
    mse = (1/p)*(np.linalg.norm(beta - beta_0)**2)
    print("MSE GAMP: ", mse)
    
    
    #BPDN
    bound = lam_psi*np.sqrt(p)
    beta_bpdn = run_BPDN(n, p, X, y, bound)

    
    for i in range(len(thresh_array)):
        beta_amp_q = quantize(beta, thresh_array[i])
        fpr_runs_03[run, i], fnr_runs_03[run, i], _ = fpr_fnr(beta_amp_q, beta_0)
        beta_bpdn_q = quantize(beta_bpdn, thresh_array[i])
        bpdn_fpr_runs_03[run, i], bpdn_fnr_runs_03[run, i], _ = fpr_fnr(beta_bpdn_q, beta_0)

fpr_amp_03 = np.average(fpr_runs_03, axis = 0)
fnr_amp_03 = np.average(fnr_runs_03, axis = 0)
fpr_bpdn_03 = np.average(bpdn_fpr_runs_03, axis=0)
fnr_bpdn_03 = np.average(bpdn_fnr_runs_03, axis=0)

plt.figure()
plt.plot(fnr_se_03, fpr_se_03, marker='x', label=r'SE', linestyle='dashed', color='blue')
plt.plot(fnr_amp_03, fpr_amp_03, marker='o', label=r'GAMP', linestyle='none', color='blue')
plt.plot(fnr_bpdn_03, fpr_bpdn_03, marker='o', label=r'BPDN', linestyle='dotted', color='red')
plt.grid(alpha=0.4)
plt.legend()
plt.ylabel('FPR')
plt.xlabel('FNR')
#tikzplotlib.save("pool_fig11b.tex")
