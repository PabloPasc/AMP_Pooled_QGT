#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 23:42:46 2023

@author: pp423
"""
import numpy as np
import math
from numba import jit
from amp_qgt import g_in_bayes, deriv_g_in_bayes
from se_qgt import mmse_new
import cvxpy as cp

@jit(nopython=True)
def norm_pdf(x, loc=0, scale=1):
    return np.exp(-((x-loc)/scale)**2/2)/(np.sqrt(2*np.pi)*scale)

@jit(nopython=True)
def cdf_scalar(x):
    return (1.0 + math.erf(x / np.sqrt(2.0))) / 2.0

@jit(nopython=True)
def cdf_vector(x):
    output = np.zeros(len(x))
    for i in range(len(x)):
        output[i]=(1.0 + math.erf(x[i] / np.sqrt(2.0))) / 2.0
    return output


'''=== Compute g_k for uniform noise and its derivative ==='''
def g_out_unif(u, y, tau_z, lim_const):
    
    #Lower-bound tau_z to avoid Div. by Zero error
    if np.abs(tau_z)<1e-9:
        tau_z = 1e-9
    sq_tz = np.sqrt(tau_z)

    U_1 = (y + lim_const - u)/sq_tz
    U_2 = (y - lim_const - u)/sq_tz
    
    I_2 = cdf_vector(U_1) - cdf_vector(U_2) + 1e-8
    diff_phi = norm_pdf(U_2) - norm_pdf(U_1)
    I_1 = u*I_2 + sq_tz*(diff_phi)
    I_3 = (u**2 + tau_z)*I_2 + 2*u*sq_tz*(diff_phi) + tau_z*(U_2*norm_pdf(U_2) - U_1*norm_pdf(U_1))
    g_k = (1/tau_z)*((I_1/I_2) - u)
    dg_k = (1/tau_z)*((I_3/(tau_z*I_2)) - (1/tau_z)*(I_1/I_2)**2 - 1)

    return g_k, dg_k

'''=== Compute E{Var(Z|Z^k, Y)} for uniform noise using MC samples ==='''
def var_z_zk_y_mc(tau_z, lim_const, exp_z2, n_mc):
    Z_k_samples = np.sqrt(exp_z2 - tau_z)*np.random.randn(n_mc)
    G_k_samples = np.random.randn(n_mc)
    Psi_k_samples = np.random.uniform(-lim_const, lim_const, n_mc)
    Y_samples = (Z_k_samples + np.sqrt(tau_z)*G_k_samples) + Psi_k_samples

    #Lower-bound tau_z to avoid Div. by Zero error
    if np.abs(tau_z)<1e-9:
        tau_z = 1e-9
    sq_tz = np.sqrt(tau_z)
    
    U_1 = (Y_samples + lim_const - Z_k_samples)/sq_tz
    U_2 = (Y_samples - lim_const - Z_k_samples)/sq_tz
    I_2 = cdf_vector(U_1) - cdf_vector(U_2)
    diff_phi = norm_pdf(U_2) - norm_pdf(U_1)
    I_1 = Z_k_samples*I_2 + sq_tz*(diff_phi)
    I_3 = (Z_k_samples**2 + tau_z)*I_2 + 2*Z_k_samples*sq_tz*(diff_phi) + tau_z*(U_2*norm_pdf(U_2) - U_1*norm_pdf(U_1))
    res = np.average((I_3/(I_2)) - (I_1/I_2)**2)
    
    return res

'''=== State evolution for QGT GAMP, under uniform noise ==='''
def state_ev_iid_gamp(delta, t, nu, lim_const, n_mc):
    exp_z2 = (nu/delta)
    #Initialization
    tau_z = 0.99*exp_z2 
    tau_q_array = []
    
    tau_z_inv = (1/tau_z)
    
    tau_q_inv = tau_z_inv*(1 - tau_z_inv*var_z_zk_y_mc(tau_z, lim_const, exp_z2, n_mc))

    for it in range(t):
        tau_q_array.append(1/tau_q_inv)
        tau_prev = tau_z
        
        #State Evolution Recursion
        #1e-10 added here to avoid sqrt of neg. value    
        tau_z = ((1/delta)*mmse_new(np.sqrt(tau_q_inv), nu)+1e-10)
        tau_z_inv = (1/tau_z)
        
        tau_q_inv = tau_z_inv*(1 - tau_z_inv*var_z_zk_y_mc(tau_z, lim_const, exp_z2, n_mc))
        
        #Stopping criteria
        if tau_z < 1e-50 or np.isnan(tau_z):
            tau_z = tau_prev
            break
        
        if (tau_z - tau_prev)**2/tau_z**2 < 1e-12:
            break
    
    #SE Performance estimates
    tau_q = (1/tau_q_inv)
    mse_pred = delta*(tau_z)
    nc_pred = 1 - (mse_pred/nu)
    return tau_z, mse_pred, nc_pred, tau_q_array, tau_q


'''=== Compute QGT GAMP, under uniform noise ==='''
def gamp_unif_noise(A, A_T, y, t, nu, x_0, lim_const):
    m, n = len(A), len(A[0])
    delta = m/n
    
    # Initialize
    p = np.zeros(m)
    tau_p_prev = 1000
    
    exp_x2 = nu
    var_x = nu - (nu)**2
    exp_x = nu
    
    tau_p = (var_x/delta)*0.99
    
    
    g_in_q = np.ones(n)*exp_x
    
    p = np.dot(A, g_in_q)
    
    g_k, dg_k = g_out_unif(p, y, tau_p, lim_const)
    
    #Set Initial tau_q to be large, positive to avoid nan error in 1st 
    #iteration (tau_p can give negative value for p very close to 0)
    tau_q = 1/np.average(-dg_k)

    # Calculate q
    q = g_in_q + tau_q*np.dot(A_T, g_k)

    error_norm_array = []
    norm_correl_array = []
    tau_array = []
    
    for it in range(t):
        
        #Estimate of x
        x_hat = g_in_q
        
        #Error might be calculated differently
        mse_x_pos = (1/n)*(np.linalg.norm(x_hat - x_0)**2)
        mse_x_neg = (1/n)*(np.linalg.norm(-x_hat - x_0)**2)
        mse = min(mse_x_pos, mse_x_neg)
        
        #ALT Measure of Performance: Normalised Correlation
        norm_correl = (np.dot(x_hat, x_0)/(np.linalg.norm(x_hat)*np.linalg.norm(x_0)))**2
        
        # End loop if nan
        if np.isnan(mse) or np.isnan(norm_correl):
            break
        
        error_norm_array.append(mse)
        norm_correl_array.append(norm_correl)
        
        
        g_in_q = g_in_bayes(q, tau_q, nu)
        print("g_in_q", g_in_q)
        
        # Calculate SE parameter tau_p - exact from q (alt: use SE)
        #tau_p = (1/delta)*mmse_new(np.sqrt(1/tau_q), nu)
        tau_p = (tau_q/m)*np.sum(deriv_g_in_bayes(q, tau_q, nu))
        print("Tau_p", tau_p)

       
        if np.isnan(tau_p):
            break
        
        # Calculate iterate p
        p = np.dot(A, g_in_q)- tau_p*g_k
        
        # Calculate g_k, dg_k for p
        g_k, dg_k = g_out_unif(p, y, tau_p, lim_const)
        
        tau_array.append(tau_q)
        
        # Calculate SE parameter tau_q
        tau_q = 1/np.average(-dg_k)
        print("Tau_q at t={}: ".format(it), tau_q)
        
        if tau_q<=1e-8 and tau_q > -1 or np.isnan(tau_q):
            break
        
        # Calculate iterate q
        q = g_in_q + tau_q*np.dot(A_T, g_k)
        
        #Stopping criterion - Relative norm tolerance
        if (tau_p - tau_p_prev)**2/tau_p**2 < 1e-18 and it >= 1:
            break
        
        tau_p_prev = 0 + tau_p
    
    return error_norm_array, norm_correl_array, x_hat, tau_array

'''=== Implement BPDN for noisy QGT ==='''
def run_BPDN(n, p, X, y, bound):
  beta_hat = cp.Variable(p) # variable to optimize
  ones_vec = np.ones(p)

  objective = cp.Minimize(ones_vec @ beta_hat) # Objective
  constraints = []
  constraints.append(beta_hat >= 0)
  constraints.append(beta_hat <= 1)
  constraints.append(cp.pnorm(y - X @ beta_hat, p='inf') <= bound)

  problem = cp.Problem(objective, constraints)

  try:
    result = problem.solve()
  except cp.error.SolverError:
    result = problem.solve(solver='SCS')

  print("...Optimal objective:", problem.value)

  return beta_hat.value


"""lam_psi = 0.3
alpha = 0.5
nu = 0.1
delta = 0.5
lim_const = lam_psi/np.sqrt(delta*alpha*(1-alpha))
#print(state_ev_iid_gamp(delta, 10, nu, lim_const, 100000))

run_no = 100
delta_array_zoom = np.linspace(0.1, 1.5, num=15)
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
lp_nc_array_av = []
lp_nc_array_std = []
se_nc_array = []
#delta_array_new = [0.8]
for delta in delta_array:
    print("Delta: ", delta)
    n = 500
    m = int(delta*n)
    
    lim_const = lam_psi/np.sqrt(delta*alpha*(1-alpha))
    
    mse_runs = []
    nc_runs = []
    nc_runs_amp = []
    mse_runs_lp = []
    nc_runs_lp = []
    
    #IID
    for run in range(run_no):
        beta_0 = create_beta(nu, n)
    
        t = 200
    
        print("Run: ", run)
        X = binomial(1, alpha, (m,n))
        psi = np.random.uniform(-lam_psi*np.sqrt(n), lam_psi*np.sqrt(n), m)
        #X = create_A_iid(m, n)
        y = np.dot(X, beta_0) + psi
        
        
        #AMP
        X_tilde = Xiid_to_Xtilde(X, alpha)
        
        #Shouldn't do this - cheating!
        #y_tilde_true = np.dot(X_tilde, beta_0)
        defect_no = np.sum(beta_0)
        
        y_tilde = y_iid_to_y_iid_tilde(y, alpha, nu, m, n, defect_no)
        
        X_tilde_T = np.transpose(X_tilde)
        #beta_amp, mse_pred_amp, tau_array_amp, error_norm_array_amp = amp_bayes(X, np.transpose(X), y, t, nu, beta_0)
        beta_amp, mse_pred_amp, tau_array_amp, error_norm_array_amp, nc_array_amp = amp_bayes(X_tilde, X_tilde_T, y_tilde, t, nu, beta_0)
        nc_amp = (np.dot(beta_amp, beta_0)/(np.linalg.norm(beta_amp)*np.linalg.norm(beta_0)))**2
        
        #beta = quantize_new(beta, thresh)
        
        error_norm_array, nc_array, beta, tau_array2 = gamp_unif_noise(X_tilde, X_tilde_T, y_tilde, t, nu, beta_0, lim_const)#gamp_iid_unif(X_tilde, X_tilde_T, y, t, lim_const, beta_0)
        
        mse = (1/n)*(np.linalg.norm(beta - beta_0)**2)
        norm_correl = (np.dot(beta, beta_0)/(np.linalg.norm(beta)*np.linalg.norm(beta_0)))**2
        
        mse_runs.append(mse)
        nc_runs.append(norm_correl)
        nc_runs_amp.append(nc_amp)
        
    mse_array_av.append(np.average(mse_runs))
    mse_array_std.append(np.std(mse_runs))
    nc_array_av.append(np.average(nc_runs))
    nc_array_std.append(np.std(nc_runs))
    nc_array_av_amp.append(np.average(nc_runs_amp))
    nc_array_std_amp.append(np.std(nc_runs_amp))
    lp_nc_array_av.append(np.average(nc_runs_lp))
    lp_nc_array_std.append(np.std(nc_runs_lp))

se_delta_array = np.linspace(0.1, 1.5, num=71)
    
for delta in se_delta_array:
    lim_const = lam_psi/np.sqrt(delta*alpha*(1-alpha))
    _, se_mse_pred, se_nc_pred, _ = state_ev_iid_gamp(delta, 200, nu, lim_const, 100000)
    se_nc_array.append(se_nc_pred)
    
plt.figure()
plt.plot(se_delta_array, se_nc_array, label=r'SE, GAMP', color = 'blue', linestyle = 'dashed')
plt.errorbar(delta_array, nc_array_av, yerr=nc_array_std, label =r"GAMP", fmt='*', color='blue',ecolor='lightblue', elinewidth=3, capsize=0)
plt.errorbar(delta_array, nc_array_av_amp, yerr=nc_array_std_amp, label =r"AMP", fmt='*', color='red',ecolor='coral', elinewidth=3, capsize=0)
plt.grid(alpha=0.4)
plt.xlabel(r'$\delta=n/p$')
plt.ylabel('Correlation')
#plt.plot(delta*np.array(tau_array2), label='tau')
#plt.plot(error_norm_array, label='Error')
plt.legend()
#tikzplotlib.save("unif_noise_lam03.tex")"""