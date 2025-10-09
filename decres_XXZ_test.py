import numpy as np
import csv
import networkx as nx
import sys, os
import random
from matplotlib import pyplot as plt
#from matplotlib import colormaps as cm
from scipy.optimize import curve_fit
from scipy.optimize import fsolve
import sys
from scipy.special import binom
#from histo_maker import *


plt.rcParams["figure.figsize"] = [6,6]
#plt.rcParams.update({"text.usetex": True, "font.family": "serif", "font.size": 14})

def powerLawDistribution(n, theta, x_min, x_max):
    number = np.array([])
    alpha = 2-theta
    for i in range(n):
        norm = x_max**(1-alpha) - x_min**(1-alpha)
        u_val = random.uniform(0, 1)
        number = np.append(number,random.choice((-1, 1)) * (u_val*norm + x_min**(1-alpha))**(1/(1-alpha)))

    return number



def bootstrap_theta(n_vec, w_values, n_bootstrap=1000, ci=68):
    """
    Compute bootstrap error bars for theta(w).

    Parameters
    ----------
    n_vec : list of 1D arrays
        Realizations of the distribution (all aligned/padded to same length).
    w_values : array
        Values of w corresponding to bins of n_vec.
    n_bootstrap : int, optional
        Number of bootstrap resamplings. Default = 1000.
    ci : float, optional
        Confidence interval width (in percent). Default = 68 (≈ 1 * sigma).

    Returns
    -------
    theta_mean : array
        Mean theta(w) over bootstrap samples.
    theta_err : array
        Error bars for theta(w), estimated from bootstrap distribution.
    """
    n_vec = np.array(n_vec)
    n_realizations, n_bins = n_vec.shape
    
    theta_boot = []
    for _ in range(n_bootstrap):
        # resample realizations with replacement
        sample_idx = np.random.randint(0, n_realizations, n_realizations)
        sample = n_vec[sample_idx]
        
        # average histogram over resampled set
        nint = np.nanmean(sample, axis=0)
        
        # compute theta
        theta_sample = 1 + np.diff(np.log(nint)) / np.diff(np.log(w_values))
        theta_boot.append(theta_sample)
    
    theta_boot = np.array(theta_boot)
    theta_mean = np.mean(theta_boot, axis=0)
    
    # error bars from central percentile range
    lower = np.percentile(theta_boot, 50 - ci/2, axis=0)
    upper = np.percentile(theta_boot, 50 + ci/2, axis=0)
    theta_err = (upper - lower) / 2
    
    return theta_mean, theta_err



cols = plt.get_cmap('viridis', 7)

dis_num = 1800

W_vec = [0.5, 1, 1.5, 2, 2.5, 3, 3.2, 3.5, 3.8, 4, 4.2, 4.4, 4.6, 4.8, 5, 5.2, 5.5, 6, 6.2, 6.5, 6.8, 7, 7.2, 7.5, 8]


for L in [14]:
    N=int(binom(L,L//2))
    for W in W_vec:

        n_vec = []

        data_histo = np.array([])  

        for dis in range(dis_num):
            
            filename_n = "Results_XXZ/niter_XXZ_Jac_L%d_W%.2f_dis%d.txt"%(L,W,dis)
            
            if os.path.isfile(filename_n):

                data_n = np.loadtxt(filename_n)
                
                n_vec.append(data_n)
                
        max_len_n = max(len(arr) for arr in n_vec)
        

        # Pad each array with NaN to make them all the same length
        padded_arrays_n = np.array([np.pad(arr, (0, max_len_n - len(arr)), 'constant', constant_values=np.nan) for arr in n_vec])
    
        
        # Compute the mean along axis 0, ignoring NaNs
        mean_n = np.nanmean(padded_arrays_n, axis=0)
    
        nint = mean_n
        
        w_start = 1.0   # Starting value
        w_end = 0.01    # Ending value
        factor = 1.1    # Geometric progression factor

        # Compute the number of steps needed
        num_steps = int(np.log(w_end / w_start) / np.log(1/1.1))

        # Generate the sequence using geomspace
        w_values = [w_start/(factor**i) for i in range(num_steps)]

        #theta = 1 + np.divide(np.diff(np.log(nint)),np.diff(np.log(w_values)))

        theta_mean, theta_err = bootstrap_theta(padded_arrays_n, w_values)
        
        filename = "Results_summary_bin/theta_XXZ_Jac_L%d_W%.2f_dis%d_test.txt"%(L,W,dis)
        np.savetxt(filename, theta_mean)

        filename = "Results_summary_bin/theta_err_XXZ_Jac_L%d_W%.2f_dis%d_test.txt"%(L,W,dis)
        np.savetxt(filename, theta_err)

        filename = "Results_summary_bin/rho_dec_XXZ_Jac_L%d_W%.2f_dis%d_test.txt"%(L,W,dis)
        np.savetxt(filename, mean_n/N)

        
