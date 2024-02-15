# AMP_Pooled_QGT
Includes code used to produce all figures in "Approximate Message Passing with Rigorous Guarantees for Pooled Data and Quantitative Group Testing" by Nelvin Tan, Pablo Pascual Cobo, Jonathan Scarlett and Ramji Venkataramanan - https://arxiv.org/abs/2309.15507.

All plots are generated using Python files.


# Required Packages
In order to run the files, the following Python libraries are required: _numpy_, _scipy_, _math_, _numba_, _cvxpy_ and _matplotlib_. In addition, library _tikzplotlib_ can be used to generate tikz files corresponding to each of the plots.

# Python Scripts
## pool_amp.py
Includes functions for matrix creation and transformations, as well as the denoising function, state evolution (SE) and approximate message passing (AMP) for pooled data testing. This script also includes functions to implement linear programming (LP), convex optimization (CVX) and iterative hard thresholding (IHT). 

## amp_qgt.py
Includes functions for matrix and vector creation and transformations, as well as the denoising function and approximate message passing (AMP) for quantitative group testing (QGT). This script also includes functions to implement linear programming (LP) for QGT.

## se_qgt.py
Includes functions to compute the corresponding state evolution (SE) to AMP QGT. 

## gamp_qgt.py
Includes output function $g_k$, functions for state evolution (SE) and generalized approximate message passing (GAMP) for QGT under uniform noise. This script also includes a function to implement basis pursuit denoising (BPDN) for noisy QGT.

## fig1_makeplots.py
Generates plots for AMP for noiseless pooled data in Fig. 1a (L=2) and Fig. 1b (L=3).

## fig2_makeplots.py
Generates AMP vs CVX, IHT plots for pooled data testing in Fig. 2a ($\sigma=0$), Fig. 2b ($\sigma=0.1$), Fig. 2c ($\sigma=0.3$).

## fig3_makeplots.py
Generates plots for AMP for pooled data under different Gaussian noise levels in Fig. 3a (L=2) and Fig. 3b (L=3), as well as plot for pooled data with mismatched proportions $\pi_{\text{est}}=[\hat{\pi}_1 + \epsilon, \hat{\pi}_2 + \epsilon]$ in Fig. 4.

## fig5_makeplots.py
Generates FPR vs FNR plots for QGT AMP and State Evolution in Fig. 5. 

## fig6_makeplots.py
Generates plots for AMP and LP QGT correlation with defective probabilities $\pi=0.1$ (Fig. 6a) and $\pi=0.3$ (Fig. 6b).

## fig7_makeplots.py
Generates plots for AMP and LP QGT FPR vs FNR with defective probabilities $\pi=0.1$ (Fig. 7a) and $\pi=0.3$ (Fig. 7b).

## fig8_makeplots.py
Generates plots for QGT under uniform noise $\Psi_i\sim \text{Uniform}[-\lambda\sqrt{p}, \lambda\sqrt{p}]$ for noise levels $\lambda=0.1$ (Fig. 8a), $\lambda=0.2$ (Fig. 8b), $\lambda=0.3$ (Fig. 8c).

## fig9_makeplots.py
Generates FPR vs FNR plot for QGT under uniform noise $\Psi_i\sim \text{Uniform}[-\lambda\sqrt{p}, \lambda\sqrt{p}]$ for noise levels $\lambda=0.1$, $\lambda=0.2$ and $\lambda=0.3$ in Fig. 9.

## fig10_makeplots.py
Generates correlation plots for AMP and LP QGT with uniform noise, with defective probabilities $\pi=0.1$ (Fig. 10a) and $\pi=0.3$ (Fig. 10b).

## fig11_makeplots.py
Generates FPR vs FNR plots for AMP and LP QGT with uniform noise, with defective probabilities $\pi=0.1$ (Fig. 11a) and $\pi=0.3$ (Fig. 11b).


