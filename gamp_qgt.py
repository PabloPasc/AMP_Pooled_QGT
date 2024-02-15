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
