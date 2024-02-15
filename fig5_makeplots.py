#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 14:56:53 2023

@author: pp423
"""

import numpy as np
import tikzplotlib
import matplotlib.pyplot as plt
from amp_qgt import amp_bayes, create_X_iid, create_beta, Xiid_to_Xtilde, y_iid_to_y_iid_tilde
from amp_qgt import fpr_fnr, g_in_bayes
from se_qgt import state_ev_iid_disc, quantize


#----------------Figure 5------------------------------------------------------
thresh_array = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
alpha = 0.5
nu = 0.3
run_no = 100

#Delta = 0.1
delta = 0.1
p = 500
n = int(delta*p)

#State Evolution
fpr_se_01 = np.ones(len(thresh_array))
fnr_se_01 = np.ones(len(thresh_array))

n_samples = 1000000
beta_0 = create_beta(nu, n_samples)
tau, mse_pred, _, tau_array_se =  state_ev_iid_disc(delta, 200, nu)

print("MSE SE: ", mse_pred)
tau_G = tau*np.random.randn(n_samples)
beta_est = g_in_bayes(beta_0 + tau_G, tau**2, nu)
for i in range(len(thresh_array)):
    beta_q = quantize(beta_est, thresh_array[i])
    fpr_se_01[i], fnr_se_01[i], _ = fpr_fnr(beta_q, beta_0)
    
#AMP
fpr_runs_01 = np.ones((run_no, len(thresh_array)))*5
fnr_runs_01 = np.ones((run_no, len(thresh_array)))*5

for run in range(run_no):
    X =  create_X_iid(alpha, n, p)
    beta_0 = create_beta(nu, p)
    y = np.dot(X, beta_0)
    
    defect_no = np.sum(beta_0)
    
    X_tilde = Xiid_to_Xtilde(X, alpha)
    X_tilde_T = np.transpose(X_tilde)
    y_tilde = y_iid_to_y_iid_tilde(y, alpha, nu, n, p, defect_no)
    beta, _, tau_array, _, _ = amp_bayes(X_tilde, X_tilde_T, y_tilde, 200, nu, beta_0)
    mse = (1/p)*(np.linalg.norm(beta - beta_0)**2)
    print("MSE AMP: ", mse)
    
    for i in range(len(thresh_array)):
        beta_amp_q = quantize(beta, thresh_array[i])
        fpr_runs_01[run, i], fnr_runs_01[run, i], _ = fpr_fnr(beta_amp_q, beta_0)

fpr_amp_01 = np.average(fpr_runs_01, axis = 0)
fnr_amp_01 = np.average(fnr_runs_01, axis = 0)

plt.figure()
plt.plot(tau_array_se, label=r'SE, $\tau$')
plt.plot(tau_array, label=r'AMP, $\tau$')
plt.legend()

#Delta = 0.3
delta = 0.3
p = 500
n = int(delta*p)

#State Evolution
fpr_se_03 = np.ones(len(thresh_array))
fnr_se_03 = np.ones(len(thresh_array))

n_samples = 1000000
beta_0 = create_beta(nu, n_samples)
tau, _, _, _ =  state_ev_iid_disc(delta, 200, nu)
tau_G = tau*np.random.randn(n_samples)
beta_est = g_in_bayes(beta_0 + tau_G, tau**2, nu)
for i in range(len(thresh_array)):
    beta_q = quantize(beta_est, thresh_array[i])
    fpr_se_03[i], fnr_se_03[i], _ = fpr_fnr(beta_q, beta_0)

#AMP
fpr_runs_03 = np.ones((run_no, len(thresh_array)))*5
fnr_runs_03 = np.ones((run_no, len(thresh_array)))*5

for run in range(run_no):
    X =  create_X_iid(alpha, n, p)
    beta_0 = create_beta(nu, p)
    y = np.dot(X, beta_0)
    
    defect_no = np.sum(beta_0)
    
    X_tilde = Xiid_to_Xtilde(X, alpha)
    X_tilde_T = np.transpose(X_tilde)
    y_tilde = y_iid_to_y_iid_tilde(y, alpha, nu, n, p, defect_no)
    beta, _, _, _, _ = amp_bayes(X_tilde, X_tilde_T, y_tilde, 200, nu, beta_0)
    
    for i in range(len(thresh_array)):
        beta_amp_q = quantize(beta, thresh_array[i])
        fpr_runs_03[run, i], fnr_runs_03[run, i], _ = fpr_fnr(beta_amp_q, beta_0)

fpr_amp_03 = np.average(fpr_runs_03, axis = 0)
fnr_amp_03 = np.average(fnr_runs_03, axis = 0)

#Delta = 0.5
delta = 0.5
p = 500
n = int(delta*p)

#State Evolution
fpr_se_05 = np.ones(len(thresh_array))
fnr_se_05 = np.ones(len(thresh_array))

n_samples = 1000000
beta_0 = create_beta(nu, n_samples)
tau, _, _, _ =  state_ev_iid_disc(delta, 200, nu)
tau_G = tau*np.random.randn(n_samples)
beta_est = g_in_bayes(beta_0 + tau_G, tau**2, nu)
for i in range(len(thresh_array)):
    beta_q = quantize(beta_est, thresh_array[i])
    fpr_se_05[i], fnr_se_05[i], _ = fpr_fnr(beta_q, beta_0)
    
#AMP
fpr_runs_05 = np.ones((run_no, len(thresh_array)))
fnr_runs_05 = np.ones((run_no, len(thresh_array)))

for run in range(run_no):
    X =  create_X_iid(alpha, n, p)
    beta_0 = create_beta(nu, p)
    y = np.dot(X, beta_0)
    
    defect_no = np.sum(beta_0)
    
    X_tilde = Xiid_to_Xtilde(X, alpha)
    X_tilde_T = np.transpose(X_tilde)
    y_tilde = y_iid_to_y_iid_tilde(y, alpha, nu, n, p, defect_no)
    beta, _, _, _, _ = amp_bayes(X_tilde, X_tilde_T, y_tilde, 200, nu, beta_0)
    
    for i in range(len(thresh_array)):
        beta_amp_q = quantize(beta, thresh_array[i])
        fpr_runs_05[run, i], fnr_runs_05[run, i], _ = fpr_fnr(beta_amp_q, beta_0)

fpr_amp_05 = np.average(fpr_runs_05, axis = 0)
fnr_amp_05 = np.average(fnr_runs_05, axis = 0)
    
    
plt.figure()
plt.plot(fnr_se_01, fpr_se_01, marker='x', label=r'SE ($\delta$=0.1)', linestyle='dashed', color='green')
plt.plot(fnr_se_03, fpr_se_03, marker='x', label=r'SE ($\delta$=0.3)', linestyle='dashed', color='red')
plt.plot(fnr_se_05, fpr_se_05, marker='x', label=r'SE ($\delta$=0.5)', linestyle='dashed', color='blue')
plt.plot(fnr_amp_01, fpr_amp_01, marker='o', label=r'AMP ($\delta$=0.1)', linestyle='none', color='green')
plt.plot(fnr_amp_03, fpr_amp_03, marker='o', label=r'AMP ($\delta$=0.3)', linestyle='none', color='red')
plt.plot(fnr_amp_05, fpr_amp_05, marker='o', label=r'AMP ($\delta$=0.5)', linestyle='none', color='blue')
plt.grid(alpha=0.4)
plt.legend()
plt.ylabel('FPR')
plt.xlabel('FNR')
#tikzplotlib.save("pool_fig5.tex")
