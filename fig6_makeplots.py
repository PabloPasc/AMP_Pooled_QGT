#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 14:56:53 2023

@author: pp423
"""

import numpy as np
import tikzplotlib
import matplotlib.pyplot as plt
from amp_qgt import amp_bayes, create_beta, Xiid_to_Xtilde, y_iid_to_y_iid_tilde
from amp_qgt import run_LP
from se_qgt import state_ev_iid_disc
from numpy.random import binomial

#----------------Figure 6a------------------------------------------------------
alpha = 0.5
nu = 0.1
run_no = 100


p = 500

delta_array = np.linspace(0.1, 1.1, num=11)

se_delta_array = np.linspace(0.1, 1.1, num=51)

nc_array_av = []
nc_array_std = []
lp_nc_array_av = []
lp_nc_array_std = []
se_nc_array = []


for delta in delta_array:
    print("Delta: ", delta)
    n = int(delta*p)
    
    mse_runs = []
    nc_runs = []
    mse_runs_lp = []
    nc_runs_lp = []
    
    #IID
    for run in range(run_no):
        beta_0 = create_beta(nu, p)
    
        t = 100
    
        alpha = 0.5
    
        print("Run: ", run)
        X = binomial(1, alpha, (n,p))
        y = np.dot(X, beta_0)
        
        #LP
        beta_lp = run_LP(n, p, X, y)
        nc_lp = (np.dot(beta_lp, beta_0)/(np.linalg.norm(beta_lp)*np.linalg.norm(beta_0)))**2
        nc_runs_lp.append(nc_lp)
        
        #AMP
        X_tilde = Xiid_to_Xtilde(X, alpha)
        
        defect_no = np.sum(beta_0)
        
        y_tilde = y_iid_to_y_iid_tilde(y, alpha, nu, n, p, defect_no)
        X_tilde_T = np.transpose(X_tilde)
        beta, mse_pred, tau_array, error_norm_array, nc_array = amp_bayes(X_tilde, X_tilde_T, y_tilde, t, nu, beta_0)
        norm_correl = (np.dot(beta, beta_0)/(np.linalg.norm(beta)*np.linalg.norm(beta_0)))**2
        
        nc_runs.append(norm_correl)
        

    nc_array_av.append(np.average(nc_runs))
    nc_array_std.append(np.std(nc_runs))
    lp_nc_array_av.append(np.average(nc_runs_lp))
    lp_nc_array_std.append(np.std(nc_runs_lp))
    
for delta in se_delta_array:    
    #IID STATE EVOLUTION
    tau, mse_pred, nc_pred = state_ev_iid_disc(delta, t, nu)
    se_nc_array.append(nc_pred)
    
    
plt.figure()
plt.plot(se_delta_array, se_nc_array, label=r'SE', color = 'blue', linestyle = 'dashed')
plt.errorbar(delta_array, nc_array_av, yerr=nc_array_std, label =r"AMP", fmt='*', color='blue',ecolor='lightblue', elinewidth=3, capsize=0)
plt.errorbar(delta_array, lp_nc_array_av, yerr=lp_nc_array_std, label =r"LP", fmt='*', color='red',ecolor='coral', elinewidth=3, capsize=0, linestyle='dotted')
plt.grid(alpha=0.4)
plt.legend()
plt.ylabel('Correlation')
plt.xlabel(r'$\delta=n/p$')
#tikzplotlib.save("pool_fig6a.tex")


#----------------Figure 6b------------------------------------------------------
alpha = 0.5
nu = 0.3
run_no = 100


p = 500

delta_array = np.linspace(0.1, 1.1, num=11)

se_delta_array = np.linspace(0.1, 1.1, num=51)

nc_array_av = []
nc_array_std = []
lp_nc_array_av = []
lp_nc_array_std = []
se_nc_array = []


for delta in delta_array:
    print("Delta: ", delta)
    n = int(delta*p)
    
    mse_runs = []
    nc_runs = []
    mse_runs_lp = []
    nc_runs_lp = []
    
    #IID
    for run in range(run_no):
        beta_0 = create_beta(nu, p)
    
        t = 100
    
        alpha = 0.5
    
        print("Run: ", run)
        X = binomial(1, alpha, (n,p))
        y = np.dot(X, beta_0)
        
        #LP
        beta_lp = run_LP(n, p, X, y)
        nc_lp = (np.dot(beta_lp, beta_0)/(np.linalg.norm(beta_lp)*np.linalg.norm(beta_0)))**2
        nc_runs_lp.append(nc_lp)
        
        #AMP
        X_tilde = Xiid_to_Xtilde(X, alpha)
        
        defect_no = np.sum(beta_0)
        
        y_tilde = y_iid_to_y_iid_tilde(y, alpha, nu, n, p, defect_no)
        X_tilde_T = np.transpose(X_tilde)
        beta, mse_pred, tau_array, error_norm_array, nc_array = amp_bayes(X_tilde, X_tilde_T, y_tilde, t, nu, beta_0)
        norm_correl = (np.dot(beta, beta_0)/(np.linalg.norm(beta)*np.linalg.norm(beta_0)))**2
        
        nc_runs.append(norm_correl)
        

    nc_array_av.append(np.average(nc_runs))
    nc_array_std.append(np.std(nc_runs))
    lp_nc_array_av.append(np.average(nc_runs_lp))
    lp_nc_array_std.append(np.std(nc_runs_lp))
    
for delta in se_delta_array:    
    #IID STATE EVOLUTION
    tau, mse_pred, nc_pred, _ = state_ev_iid_disc(delta, t, nu)
    se_nc_array.append(nc_pred)
    
    
plt.figure()
plt.plot(se_delta_array, se_nc_array, label=r'SE', color = 'blue', linestyle = 'dashed')
plt.errorbar(delta_array, nc_array_av, yerr=nc_array_std, label =r"AMP", fmt='*', color='blue',ecolor='lightblue', elinewidth=3, capsize=0)
plt.errorbar(delta_array, lp_nc_array_av, yerr=lp_nc_array_std, label =r"LP", fmt='*', color='red',ecolor='coral', elinewidth=3, capsize=0, linestyle='dotted')
plt.grid(alpha=0.4)
plt.legend()
plt.ylabel('Correlation')
plt.xlabel(r'$\delta=n/p$')
#tikzplotlib.save("pool_fig6b.tex")