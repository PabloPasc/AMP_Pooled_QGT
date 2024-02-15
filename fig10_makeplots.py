#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 23:27:47 2023

@author: pp423
"""

import numpy as np
from amp_qgt import Xiid_to_Xtilde, y_iid_to_y_iid_tilde, create_beta
from gamp_qgt import gamp_unif_noise, state_ev_iid_gamp, run_BPDN
from numpy.random import binomial
import matplotlib.pyplot as plt

#--------------------------------Fig.10a---------------------------------------
lam_psi = 0.1
alpha = 0.5
nu = 0.1

run_no = 100
delta_array_zoom = np.linspace(0.1, 1.0, num=10)
delta_array = np.sort(np.concatenate([[0.95, 1.05, 1.15],delta_array_zoom]))
mse_array_av = []
mse_array_std = []
lp_mse_array_av = []
lp_mse_array_std = []
se_mse_array = []
sc_mse_array = []
sc_se_mse_array = []
nc_array_av = []
nc_array_std = []
nc_array_av_amp = []
nc_array_std_amp = []
bpdn_nc_array_av = []
bpdn_nc_array_std = []
se_nc_array = []

for delta in delta_array_zoom:
    print("Delta: ", delta)
    n = 500
    m = int(delta*n)
    
    lim_const = lam_psi/np.sqrt(delta*alpha*(1-alpha))
    bound = lam_psi*np.sqrt(n)
    
    mse_runs = []
    nc_runs = []
    nc_runs_amp = []
    mse_runs_bpdn = []
    nc_runs_bpdn = []
    
    #IID
    for run in range(run_no):
        beta_0 = create_beta(nu, n)
    
        t = 200
    
        print("Run: ", run)
        X = binomial(1, alpha, (m,n))
        psi = np.random.uniform(-bound, bound, m)

        y = np.dot(X, beta_0) + psi
        
        
        #GAMP
        X_tilde = Xiid_to_Xtilde(X, alpha)

        defect_no = np.sum(beta_0)
        
        y_tilde = y_iid_to_y_iid_tilde(y, alpha, nu, m, n, defect_no)
        
        X_tilde_T = np.transpose(X_tilde)

        error_norm_array, nc_array, beta, tau_array2 = gamp_unif_noise(X_tilde, X_tilde_T, y_tilde, t, nu, beta_0, lim_const)#gamp_iid_unif(X_tilde, X_tilde_T, y, t, lim_const, beta_0)
        
        mse = (1/n)*(np.linalg.norm(beta - beta_0)**2)
        norm_correl = (np.dot(beta, beta_0)/(np.linalg.norm(beta)*np.linalg.norm(beta_0)))**2
        
        mse_runs.append(mse)
        nc_runs.append(norm_correl)
        
        #BPDN
        beta_bpdn = run_BPDN(m, n, X, y, bound)
        nc_bpdn = (np.dot(beta_bpdn, beta_0)/(np.linalg.norm(beta_bpdn)*np.linalg.norm(beta_0)))**2
        
        nc_runs_bpdn.append(nc_bpdn)
        
        
    mse_array_av.append(np.average(mse_runs))
    mse_array_std.append(np.std(mse_runs))
    nc_array_av.append(np.average(nc_runs))
    nc_array_std.append(np.std(nc_runs))
    nc_array_av_amp.append(np.average(nc_runs_amp))
    nc_array_std_amp.append(np.std(nc_runs_amp))
    bpdn_nc_array_av.append(np.average(nc_runs_bpdn))
    bpdn_nc_array_std.append(np.std(nc_runs_bpdn))

se_delta_array = np.linspace(0.1, 1, num=46)
    
for delta in se_delta_array:
    lim_const = lam_psi/np.sqrt(delta*alpha*(1-alpha))
    _, se_mse_pred, se_nc_pred, _, _ = state_ev_iid_gamp(delta, 200, nu, lim_const, 100000)
    se_nc_array.append(se_nc_pred)
    
plt.figure()
plt.plot(se_delta_array, se_nc_array, label=r'SE, GAMP', color = 'blue', linestyle = 'dashed')
plt.errorbar(delta_array_zoom, nc_array_av, yerr=nc_array_std, label =r"GAMP", fmt='*', color='blue',ecolor='lightblue', elinewidth=3, capsize=0)
plt.errorbar(delta_array_zoom, bpdn_nc_array_av, yerr=bpdn_nc_array_std, label =r"BPDN", fmt='*', color='red',ecolor='coral', elinewidth=3, capsize=0, linestyle='dotted')
plt.grid(alpha=0.4)
plt.xlabel(r'$\delta=n/p$')
plt.ylabel('Correlation')
plt.legend()
#tikzplotlib.save("pool_fig10a.tex")


#----------------------Fig. 10b------------------------------------------------
lam_psi = 0.1
alpha = 0.5
nu = 0.3

run_no = 100
delta_array_zoom = np.linspace(0.1, 1.0, num=10)
delta_array = np.sort(np.concatenate([[0.95, 1.05, 1.15],delta_array_zoom]))
mse_array_av = []
mse_array_std = []
lp_mse_array_av = []
lp_mse_array_std = []
se_mse_array = []
sc_mse_array = []
sc_se_mse_array = []
nc_array_av = []
nc_array_std = []
nc_array_av_amp = []
nc_array_std_amp = []
bpdn_nc_array_av = []
bpdn_nc_array_std = []
se_nc_array = []

for delta in delta_array_zoom:
    print("Delta: ", delta)
    n = 500
    m = int(delta*n)
    
    lim_const = lam_psi/np.sqrt(delta*alpha*(1-alpha))
    bound = lam_psi*np.sqrt(n)
    
    mse_runs = []
    nc_runs = []
    nc_runs_amp = []
    mse_runs_bpdn = []
    nc_runs_bpdn = []
    
    #IID
    for run in range(run_no):
        beta_0 = create_beta(nu, n)
    
        t = 200
    
        print("Run: ", run)
        X = binomial(1, alpha, (m,n))
        psi = np.random.uniform(-bound, bound, m)

        y = np.dot(X, beta_0) + psi
        
        
        #GAMP
        X_tilde = Xiid_to_Xtilde(X, alpha)

        defect_no = np.sum(beta_0)
        
        y_tilde = y_iid_to_y_iid_tilde(y, alpha, nu, m, n, defect_no)
        
        X_tilde_T = np.transpose(X_tilde)
        
        error_norm_array, nc_array, beta, tau_array2 = gamp_unif_noise(X_tilde, X_tilde_T, y_tilde, t, nu, beta_0, lim_const)#gamp_iid_unif(X_tilde, X_tilde_T, y, t, lim_const, beta_0)
        
        mse = (1/n)*(np.linalg.norm(beta - beta_0)**2)
        norm_correl = (np.dot(beta, beta_0)/(np.linalg.norm(beta)*np.linalg.norm(beta_0)))**2
        
        mse_runs.append(mse)
        nc_runs.append(norm_correl)
        
        #BPDN
        beta_bpdn = run_BPDN(m, n, X, y, bound)
        nc_bpdn = (np.dot(beta_bpdn, beta_0)/(np.linalg.norm(beta_bpdn)*np.linalg.norm(beta_0)))**2
        
        nc_runs_bpdn.append(nc_bpdn)
        
        
    mse_array_av.append(np.average(mse_runs))
    mse_array_std.append(np.std(mse_runs))
    nc_array_av.append(np.average(nc_runs))
    nc_array_std.append(np.std(nc_runs))
    nc_array_av_amp.append(np.average(nc_runs_amp))
    nc_array_std_amp.append(np.std(nc_runs_amp))
    bpdn_nc_array_av.append(np.average(nc_runs_bpdn))
    bpdn_nc_array_std.append(np.std(nc_runs_bpdn))

se_delta_array = np.linspace(0.1, 1, num=46)
    
for delta in se_delta_array:
    lim_const = lam_psi/np.sqrt(delta*alpha*(1-alpha))
    _, se_mse_pred, se_nc_pred, _, _ = state_ev_iid_gamp(delta, 200, nu, lim_const, 100000)
    se_nc_array.append(se_nc_pred)
    
plt.figure()
plt.plot(se_delta_array, se_nc_array, label=r'SE, GAMP', color = 'blue', linestyle = 'dashed')
plt.errorbar(delta_array_zoom, nc_array_av, yerr=nc_array_std, label =r"GAMP", fmt='*', color='blue',ecolor='lightblue', elinewidth=3, capsize=0)
plt.errorbar(delta_array_zoom, bpdn_nc_array_av, yerr=bpdn_nc_array_std, label =r"BPDN", fmt='*', color='red',ecolor='coral', elinewidth=3, capsize=0, linestyle='dotted')
plt.grid(alpha=0.4)
plt.xlabel(r'$\delta=n/p$')
plt.ylabel('Correlation')
plt.legend()
#tikzplotlib.save("pool_fig10b.tex")