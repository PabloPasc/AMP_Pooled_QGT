# AMP_Pooled_QGT
Includes code used to produce all figures in "Approximate Message Passing with Rigorous Guarantees for Pooled Data and Quantitative Group Testing" by Nelvin Tan, Pablo Pascual Cobo, Jonathan Scarlett and Ramji Venkataramanan - https://arxiv.org/abs/2309.15507.

All plots are generated using Python files.


# Required Packages
In order to run the files, the following Python libraries are required: _numpy_, _scipy_, _numba_, _cvxpy_ and _matplotlib_. In addition, library _tikzplotlib_ can be used to generate tikz files corresponding to each of the plots.

# Python Scripts
## fig1_makeplots.py
Generates plots for AMP for noiseless pooled data in Fig. 1a (L=2) and Fig. 1b (L=3).

## fig2_makeplots.py
Generates AMP vs CVX, IHT plots for pooled data testing in Fig. 2a ($\sigma=0$), Fig. 2b ($\sigma=0.1$), Fig. 2c ($\sigma=0.3$).

## fig3_makeplots.py
Generates plots for AMP for pooled data under different Gaussian noise levels in Fig. 3a (L=2) and Fig. 3b (L=3).

## fig4_makeplots.py
Generates plot for pooled data with mismatched proportions $\pi_{\text{est}}=[\hat{\pi}_1 + \epsilon, \hat{\pi}_2 + \epsilon]$ in Fig. 4.


