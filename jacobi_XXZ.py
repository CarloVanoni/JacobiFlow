import numpy as np
import numba
import sys, os
from matplotlib import pyplot as plt
import random as rnd
import random
import networkx as nx
from numba import njit
from numba import jit
from numpy.random import default_rng
from scipy.sparse import lil_matrix
from scipy.linalg import expm
from functools import reduce
import scipy.sparse as sps
from scipy import sparse
from scipy.special import binom
#from histo_maker import *
import scipy
import time
from numba.typed import List
from scipy.stats import bernoulli, lognorm




rng = default_rng()

@njit
def compute_r(data):
    """
    Compute the ratio r of adjacent level spacings for a subset of the sorted data.

    Parameters:
    - data: 1D array-like of eigenvalues or energy levels.

    Returns:
    - r: array of ratios min(s_i, s_{i+1}) / max(s_i, s_{i+1}) computed over a central window.
    
    Notes:
    - This uses a window centered around N//2 with size 50 (from N//2-25 to N//2+25).
    - The global variable N is referenced here (as in the original code).
    """
    data_n = np.sort(data)
    diff = np.diff(data_n[N//2-25:N//2+25])
    return np.minimum(diff[:-1], diff[1:]) / np.maximum(diff[:-1], diff[1:])

@jit(nopython=True)
def rand_choice_nb(arr, prob):
    """
    Sample two random elements from a 1D array `arr` with probabilities `prob`.

    Parameters:
    - arr: 1D numpy array of values to sample from.
    - prob: 1D numpy array of probabilities (should sum to 1).

    Returns:
    - tuple (sample1, sample2): two sampled values from `arr`.
    """
    cumsum_prob = np.cumsum(prob)  # pre-calculate cumulative sum
    #rand_nums = np.random.random(2)  # generate two random numbers once
    idx1 = np.searchsorted(cumsum_prob, np.random.random(), side="right")
    idx2 = np.searchsorted(cumsum_prob, np.random.random(), side="right")
    return arr[idx1], arr[idx2]


@numba.jit(nopython=True)
def get_bin_edges(a, bins):
    """
    Compute uniformly spaced bin edges for array `a` with `bins` bins.

    Parameters:
    - a: 1D array-like input data.
    - bins: integer number of bins.

    Returns:
    - bin_edges: array of length bins+1 with edges from min(a) to max(a).
    """
    bin_edges = np.zeros((bins+1,), dtype=np.float64)
    a_min = a.min()
    a_max = a.max()
    delta = (a_max - a_min) / bins
    for i in range(bin_edges.shape[0]):
        bin_edges[i] = a_min + i * delta

    bin_edges[-1] = a_max  # Avoid roundoff error on last point
    return bin_edges


@numba.jit(nopython=True)
def compute_bin(x, bin_edges):
    """
    Compute the bin index for scalar `x` given precomputed uniform `bin_edges`.

    Parameters:
    - x: scalar value.
    - bin_edges: array of bin edges (length bins+1).

    Returns:
    - bin index (int) in [0, bins-1] or None if out of range.
    """
    # assuming uniform bins for now
    n = bin_edges.shape[0] - 1
    a_min = bin_edges[0]
    a_max = bin_edges[-1]

    # special case to mirror NumPy behavior for last bin
    if x == a_max:
        return n - 1 # a_max always in last bin

    bin = int(n * (x - a_min) / (a_max - a_min))

    if bin < 0 or bin >= n:
        return None
    else:
        return bin


@numba.jit(nopython=True)
def numba_histogram(a, bins):
    """
    Fast histogram implementation using numba.

    Parameters:
    - a: 1D input array
    - bins: number of histogram bins

    Returns:
    - hist: counts per bin (length `bins`)
    - bin_edges: edges array (length `bins+1`)
    """
    hist = np.zeros((bins,), dtype=np.intp)
    bin_edges = get_bin_edges(a, bins)

    for x in a.flat:
        bin = compute_bin(x, bin_edges)
        if bin is not None:
            hist[int(bin)] += 1

    return hist, bin_edges


@numba.jit(nopython=True)
def f_V(omega,omega_0,sigma_omega):
    """
    Spectral weight function f_V(omega) used in FGR-type Hamiltonian construction.

    Parameters:
    - omega: energy difference
    - omega_0: center frequency
    - sigma_omega: width of the Gaussian components

    Returns:
    - value: scalar spectral weight computed as the average of two Gaussians centered at ±omega_0.
    """
    return 1/(2*np.sqrt(2*np.pi*sigma_omega**2)) * ( np.exp(-((omega-omega_0)**2)/(2*sigma_omega**2) ) + np.exp(-((omega+omega_0)**2)/(2*sigma_omega**2) ))

#@numba.jit(nopython=True)
def off_diagonal_elements(matrix, n):
    """
    Randomly sample `n` off-diagonal elements from `matrix` along with their diagonal differences.

    Parameters:
    - matrix: square 2D numpy array
    - n: number of off-diagonal elements to sample

    Returns:
    - off_diagonal: 1D array of sampled off-diagonal elements (length n)
    - diagonal_diff: 1D array of corresponding diagonal differences H[i,i] - H[j,j] (length n)
    """
    #random.seed(42)  # For reproducibility
    indices = [(i, j) for i in range(len(matrix)) for j in range(i+1,len(matrix))]
    random.shuffle(indices)
    
    selected_indices = indices[:n]
    
    off_diagonal = np.zeros(n)
    diagonal_diff = np.zeros(n)
    
    k = 0
    for i, j in selected_indices:
        off_diagonal[k] = matrix[i][j]
        diagonal_diff[k] = matrix[i][i] - matrix[j][j]

        k+=1

    return off_diagonal, diagonal_diff


def powerLawDistribution(n, theta, x_min, x_max):
    """
    Generate `n` values drawn from a signed power-law-like distribution.

    Parameters:
    - n: number of values to generate
    - theta: parameter that controls the exponent (alpha = 2 - theta)
    - x_min, x_max: bounds used in the generation (positive)

    Returns:
    - number: numpy array of length n with values drawn from the distribution (including random ± signs)
    """
    number = np.array([])
    alpha = 2-theta
    for i in range(n):
        norm = x_max**(1-alpha) - x_min**(1-alpha)
        u_val = random.uniform(0, 1)
        number = np.append(number,random.choice((-1, 1)) * (u_val*norm + x_min**(1-alpha))**(1/(1-alpha)))

    return number


def create_upper_matrix(values, size):
    """
    Fill the strict upper-triangular part of a size x size matrix with a flat list/array of values.

    Parameters:
    - values: 1D array of length size*(size-1)/2 corresponding to upper-triangular entries
    - size: dimension of the square matrix

    Returns:
    - upper: size x size numpy array with upper triangle (excluding diagonal) filled
    """
    upper = np.zeros((size, size))
    upper[np.triu_indices(size, 1)] = values
    return(upper)


@numba.jit(nopython=True)
def build_FGR_H(N,sigma_E,omega_0,sigma_omega,J):
    """
    Build a random Hamiltonian with FGR-inspired off-diagonal entries.

    Parameters:
    - N: matrix dimension
    - sigma_E: scale for diagonal disorder (uniform range)
    - omega_0, sigma_omega: parameters for the spectral weight f_V
    - J: overall coupling strength prefactor

    Returns:
    - H: NxN symmetric matrix (numpy array)
    - off_val_tot: sum of squares of off-diagonal elements times 2 (as in original code)
    """
    off_val_tot = 0
    H = np.zeros((N,N))

    for i in range(N):
        H[i,i] = np.random.uniform(-sigma_E/2,sigma_E/2)
        #H[i,i] = np.random.normal(0,sigma_E/4)

    for i in range(N):
        for j in range(i+1,N):
            H[i,j] = J*np.random.normal(0,1)* np.sqrt(f_V(H[i,i]-H[j,j], omega_0, sigma_omega))/np.sqrt(N/sigma_E)
            H[j,i] = H[i,j]
            off_val_tot += 2*(H[j,i]**2)
    
    return H, off_val_tot

@numba.jit(nopython=True)
def build_FGR_H_box(N,sigma_E,omega_0,sigma_omega,J):
    """
    Build a random Hamiltonian similar to build_FGR_H but using uniform random off-diagonal signs (box distribution).

    Parameters:
    - N: matrix dimension
    - sigma_E: scale for diagonal disorder (uniform range)
    - omega_0, sigma_omega: parameters for the spectral weight f_V
    - J: overall coupling strength prefactor

    Returns:
    - H: NxN symmetric matrix (numpy array)
    - off_val_tot: sum of squares of off-diagonal elements times 2
    """
    off_val_tot = 0
    H = np.zeros((N,N))

    for i in range(N):
        H[i,i] = np.random.uniform(-sigma_E/2,sigma_E/2)

    for i in range(N):
        for j in range(i+1,N):
            H[i,j] = J*np.random.uniform(-1,1)* np.sqrt(f_V(H[i,i]-H[j,j], omega_0, sigma_omega))/np.sqrt(N/sigma_E)
            H[j,i] = H[i,j]
            off_val_tot += 2*(H[j,i]**2)
    
    return H, off_val_tot


def build_powlaw_H(N,theta):
    """
    Build a symmetric matrix with off-diagonal entries drawn from the custom powerLawDistribution.

    Parameters:
    - N: matrix dimension
    - theta: parameter controlling the power-law exponent

    Returns:
    - H: NxN symmetric matrix (numpy array)
    - off_val_tot: 2 * sum(r^2) where r are the raw off-diagonal upper-triangle values
    """
    r = powerLawDistribution(N*(N-1)//2,theta,0.001,1)
    off_val_tot = 2*np.sum(r**2)

    # Fills the upper triangular matrix
    c = create_upper_matrix(r, N)

    #Diagonal
    D = np.random.uniform(-2,2,N)
    H = c + c.T + np.diag(D)

    return H, off_val_tot


def build_GOE_H(N,t):
    """
    Build a GOE-like random matrix with additional large random entries controlled by t.

    Parameters:
    - N: matrix dimension
    - t: integer controlling the number of large random injections

    Returns:
    - H: symmetric NxN numpy array
    - off_val_tot: sum of squares of off-diagonal elements
    """
    H = rng.standard_normal((N,N))/np.sqrt(2*N)
    big_inds = rng.integers(N, size=N*t)
    H[np.repeat(np.arange(N),t),big_inds] += rng.standard_normal(N*t)/np.sqrt(2*t)
    H += H.T

    off_val_tot = np.sum(np.square(H)) - np.sum(np.square(np.diag(H)))

    return H, off_val_tot


@numba.jit(nopython=True)
def plbrm(N,a,b,W):
    """
    Build a Power-Law Banded Random Matrix (PLBRM) ensemble sample.

    Parameters:
    - N: matrix size
    - a, b: parameters controlling the power-law decay of variances
    - W: diagonal variance scale

    Returns:
    - init: list of off-diagonal elements appended row by row
    - H: full NxN symmetric matrix
    """
    H = np.zeros((N,N))
    init = []

    for i in range(N):
        H[i,i] = np.random.normal(0,W)
        for j in range(i):
            H[i,j] = np.random.normal(0,np.sqrt(0.5/(1+((np.abs(i-j)/b)**(2*a)))))
            H[j,i] = H[i,j]
            init.append(H[i,j])
    return init, H


def LNRP_Matrix(n,gamma,p):
    """
    Build a log-normal random matrix (LNRP) with off-diagonal lognormal-like entries and uniform diagonal.

    Parameters:
    - n: matrix size
    - gamma, p: parameters controlling the lognormal scale for off-diagonals

    Returns:
    - H: nxn symmetric matrix (numpy array)
    """
    # Extract LN off diag elements
    r = n**(-gamma/2)*np.multiply((1-2*np.random.randint(0,2,size=n*(n-1)//2)) , lognorm.rvs(np.sqrt(gamma * p * np.log(n) / 2.), size=n*(n-1)//2))

    # Fills the upper triangular matrix
    c = create_upper_matrix(r, n)

    #Diagonal
    D = rng.uniform(-1,1,size=n)

    # Full matrix
    H = c + c.T + np.diag(D)

    return H

def random_power_law(mu, N, gamma, n):
    """
    Generate n random numbers distributed according to
    p(x) = (mu / (2 * N**gamma * x**(1 + mu))) * HeavisideTheta(|x| > N**(-gamma / mu)).

    Parameters:
    - mu (float): Shape parameter of the power-law distribution.
    - N (float): A normalization or scale factor.
    - gamma (float): Controls the threshold for the truncation.
    - n (int): Number of random values to generate.

    Returns:
    - x: numpy.ndarray of length n with signed power-law distributed entries.
    """
    
    # Threshold for |x| based on the given condition
    x_min = N**(-gamma / mu)
    
    # Generate uniform random values for inverse transform sampling
    u = np.random.uniform(size=n)
    
    # Inverse CDF of the power-law distribution to get the absolute values of x
    # x_abs = (x_min) * (1 - u)**(-1/mu)
    x_abs = x_min * (1 - u)**(-1 / mu)
    
    # Randomly assign signs (positive or negative) to the generated values
    signs = np.random.choice([-1, 1], size=n)
    x = signs * x_abs
    
    return x


def Levy_Matrix(n,gamma,mu,W):
    """
    Construct a symmetric matrix with Lévy (power-law) distributed off-diagonals.

    Parameters:
    - n: matrix size
    - gamma, mu: distribution parameters passed to random_power_law
    - W: diagonal uniform range parameter

    Returns:
    - H: symmetric nxn numpy array
    """
    # Extract LN off diag elements
    r = random_power_law(mu, n, gamma, n*(n-1)//2)

    # Fills the upper triangular matrix
    c = create_upper_matrix(r, n)

    #Diagonal
    D = rng.uniform(-W/2,W/2,size=n)

    # Full matrix
    H = c + c.T + np.diag(D)

    return H


def Lbit_LN(L,eps,x,W):
    """
    Construct a hierarchical (kron-like) unitary-like operator built from small lognormal matrices.

    Parameters:
    - L: number of hierarchical steps
    - eps: scaling parameter used in expm
    - x: parameter controlling the lognormal mean/variance
    - W: scale for on-site Z terms added afterwards

    Returns:
    - M: dense matrix result after hierarchical composition and added Z terms
    """
    X=np.array([[0,1],[1,0]])
    Z=np.array([[1,0],[0,-1]])
    Id=np.array([[1,0],[0,1]])

    M = X

    for il, l in enumerate(range(1,L+2,2)):
        
        N0 = 2**l

        print(l)
        """H = np.random.normal(0,1,size=(N0,N0))
        H = (H + H.T)/2"""

        r = np.random.lognormal(mean=-l*x, sigma=np.sqrt(l*x),size=N0*(N0-1)//2)

        c = create_upper_matrix(r, N0)

        #Diagonal
        D = rng.uniform(0,0,size=N0)

        # Full matrix
        H = c + c.T + np.diag(D)
        norm = np.sum(np.square(np.abs(H)))
        #print(norm)
        
        H /= norm
        #eps = l**(-x)
        
        """U = expm(-1j*H*(eps**il))
        Udag = expm(1j*H*(eps**il))"""
        U = expm(-1j*H*(eps)**il)
        Udag = expm(1j*H*(eps)**il)

        M=np.matmul(np.matmul(Udag,M),U)

        if l < L:
            M = reduce(lambda x, y: sps.kron(x,y,format='csc'), [Id]+[M]+[Id]).todense()
        
    for i in range(L+1):
        M += reduce(lambda x, y: sps.kron(x,y,format='csc'), [Id]*i+[(2*np.random.rand()-1)*W*Z]+[Id]*(L-i)).todense()
        
    return M

def H0_XXZ(L):
    """
    Load a precomputed OFFDIAG file for XXZ model and build a projected sparse H0 matrix.

    Parameters:
    - L: system size parameter used to construct filename

    Returns:
    - H0: sparse lil_matrix after projection with Proj (Proj must be available in the environment)
    """
    filename = "XXZ_OFFDIAG_L%d.txt" %L
    data = np.loadtxt(filename)


    H0 = sparse.lil_matrix((N,N),dtype=complex)
    for i in range(len(data)):
        H0[int(data[i,0]),int(data[i,1])] = data[i,2]
    H0 = Proj @ H0 @ Proj.T

    return H0

def H_RRG(N, W):
    """
    Build a random-regular-graph adjacency matrix with added diagonal disorder.

    Parameters:
    - N: number of nodes (matrix dimension)
    - W: uniform diagonal disorder range width

    Returns:
    - H: dense NxN numpy array with adjacency + diagonal disorder
    """
    G = nx.random_regular_graph(3,N)

    H = nx.adjacency_matrix(G, dtype=float)

    H = H.todense()

    for i in range(N):
        H[i,i] = np.random.uniform(-W/2,W/2)

    return H


def fill_dist_RRG(N):
    """
    Generate pairwise shortest-path distance matrix for a random regular graph of degree 3.

    Parameters:
    - N: number of nodes

    Returns:
    - dist: NxN numpy array of graph distances
    """
    G = nx.random_regular_graph(3,N)

    dist = np.zeros((N,N))

    for i in range(N):
        for j in range(i+1,N):
            dist[i,j] = nx.shortest_path_length(G, i, j)
            dist[j,i] = dist[i,j]

    return dist


def fill_Bethe(N):
    """
    Load adjacency and compute pairwise distances for a Bethe-like tree constructed from file.

    Parameters:
    - N: input intended matrix size (function computes L from N, reassigns N internally)

    Returns:
    - dist: NxN numpy array of graph distances for the loaded Bethe graph
    """
    L = int(np.log(N)/np.log(2))
    l = 2
    N = ((l+1)*l**(L) - 2) // (l - 1)

    adj = lil_matrix((N, N))

    filename = "Adjacency/AdM_L%d.txt"%L
    data = np.loadtxt(filename).T

    adj[data[0],data[1]]=1
    adj[data[1],data[0]]=1

    G = nx.from_scipy_sparse_array(adj)

    dist = np.zeros((N,N))

    for i in range(N):
        for j in range(i+1,N):
            dist[i,j] = nx.shortest_path_length(G, i, j)
            dist[j,i] = dist[i,j]
    
    return dist

@numba.jit(nopython=True)
def fill_RRG_Jac(N,dist,w0,z,W):
    """
    Fill a Hamiltonian matrix for RRG-like connectivity for Jacobi experiments.

    Parameters:
    - N: matrix size
    - dist: NxN matrix of distances (graph distances)
    - w0, z: parameters for exponential decay prefactor of off-diagonals
    - W: diagonal disorder amplitude

    Returns:
    - H: symmetric NxN numpy array
    - off_val_tot: sum of off-diagonal squared elements
    """
    H = np.zeros((N,N))

    for i in range(N):
        for j in range(i+1,N):
            
            H[i,j] = w0*np.exp(-dist[i,j]/z)*np.random.normal(0,1)/np.sqrt(2**dist[i,j])
            H[j,i] = H[i,j]

    for i in range(N):
        H[i,i] = np.random.uniform(-W/2,W/2)

    off_val_tot = np.sum(np.square(H)) - np.sum(np.square(np.diag(H)))

    return H, off_val_tot


@numba.jit(nopython=True)
def fill_RRG_SRA(N,dist,w0,z,W):
    """
    Same as fill_RRG_Jac but also returns a flattened list of off-diagonal elements (init).

    Parameters:
    - N, dist, w0, z, W: same as fill_RRG_Jac

    Returns:
    - init: list of off-diagonal elements in the order they were created
    - H: symmetric NxN numpy array
    - off_val_tot: sum of off-diagonal squared elements
    """
    H = np.zeros((N,N))
    init = []

    for i in range(N):
        for j in range(i+1,N):
            
            H[i,j] = w0*np.exp(-dist[i,j]/z)*np.random.normal(0,1)/np.sqrt(2**dist[i,j])
            H[j,i] = H[i,j]
            init.append(H[i,j])

    for i in range(N):
        H[i,i] = np.random.uniform(-W/2,W/2)

    off_val_tot = np.sum(np.square(H)) - np.sum(np.square(np.diag(H)))

    return init, H, off_val_tot


@numba.jit(nopython=True)
def compute_dh(off_diag):
    """
    Compute mean absolute difference between consecutive sorted off-diagonal values.

    Parameters:
    - off_diag: 1D array of off-diagonal elements

    Returns:
    - mean absolute difference of sorted off_diag differences
    """
    return np.mean(np.abs(np.diff(np.sort(off_diag))))


@numba.jit(nopython=True)
def off_diag_SRA_FGR(N,n,sigma_E,omega_0,sigma_omega,J):
    """
    Sample off-diagonal elements for SRA/FGR model and compute summary statistics.

    Parameters:
    - N: size used for some normalizations
    - n: number of samples
    - sigma_E, omega_0, sigma_omega, J: model parameters

    Returns:
    - off_diag: sampled off-diagonal values (length n)
    - diag_diff: sampled diagonal differences (length n)
    - dh: mean absolute spacing of sorted off-diagonals
    - off_val_tot: sum of squares of off-diagonals
    - level_spacing: typical level spacing estimated from a random diagonal ensemble
    """
    diag_diff = np.zeros(n)
    off_diag = np.zeros(n)
    for i in range(n):
        #w = np.random.uniform(-sigma_E/2,sigma_E/2)
        #w = np.random.normal(0,sigma_E/4)
        #w2 = np.random.normal(0,sigma_E/4)
        #w1 = np.random.normal(0,sigma_E/4)
        #w2 = np.random.uniform(-sigma_E/2,sigma_E/2)
        #w1 = np.random.uniform(-sigma_E/2,sigma_E/2)

        w1 = np.random.uniform(-5*sigma_omega,5*sigma_omega)
        diag_diff[i] = w1
        #diag_diff[i] = w
        off_diag[i] = J*np.random.normal(0,1)* np.sqrt(f_V(diag_diff[i], omega_0, sigma_omega))/np.sqrt(N/sigma_E)

    dh = compute_dh(off_diag)
    off_val_tot = np.sum(np.square(off_diag))

    w_vec = np.random.normal(0,sigma_E/4,N)
    diag = np.sort(w_vec)
    level_spacing = np.abs(np.mean(np.diff(diag)))

    return off_diag,diag_diff,dh,off_val_tot,level_spacing


@numba.jit(nopython=True)
def off_diag_SRA_FGR_box(n,sigma_E,omega_0,sigma_omega,J):
    """
    Variant of off_diag_SRA_FGR using box-distributed random off-diagonals.

    Parameters:
    - n: number of samples
    - sigma_E, omega_0, sigma_omega, J: parameters

    Returns:
    - off_diag, diag_diff, dh, off_val_tot
    """
    diag_diff = np.zeros(n-1)
    off_diag = np.zeros(n-1)
    for i in range(n-1):
        w1 = np.random.uniform(-sigma_E/2,sigma_E/2)
        w2 = np.random.uniform(-sigma_E/2,sigma_E/2)
        diag_diff[i] = w1-w2
        off_diag[i] = J*np.random.uniform(-1,1)* np.sqrt(f_V(w1-w2, omega_0, sigma_omega))/np.sqrt(n/sigma_E)

    dh = compute_dh(off_diag)
    off_val_tot = np.sum(np.square(off_diag))

    return off_diag,diag_diff,dh,off_val_tot

def off_diag_SRA_powlaw(n,theta):
    """
    Produce off-diagonal samples using the custom power law for amplitudes.

    Parameters:
    - n: number of samples
    - theta: power-law parameter

    Returns:
    - off_diag, diag_diff, dh, off_val_tot
    """
    diag_diff = np.zeros(n)
    off_diag = np.zeros(n)
    for i in range(n):
        w1 = np.random.uniform(-2,2)
        w2 = np.random.uniform(-2,2)
        diag_diff[i] = w1-w2
        off_diag[i] =powerLawDistribution(1,theta,0.001,1)
    
    dh = compute_dh(off_diag)
    off_val_tot = np.sum(np.square(off_diag))

    return off_diag,diag_diff,dh,off_val_tot

@njit
def prob_from_matrix(H,N,n):
    """
    Randomly sample `n` off-diagonal entries from matrix H and return them with diagonal differences.

    Parameters:
    - H: square matrix
    - N: size of matrix
    - n: number of samples (< N typically)

    Returns:
    - off_diag: array of sampled off-diagonal entries
    - diag_diff: array of corresponding diagonal differences H[i,i] - H[j,j]
    """
    # n = population size, must be < N
    
    off_diag = np.zeros(n)
    diag_diff = np.zeros(n)

    for k in range(n):
        i1 = np.random.randint(0,N-1)
        i2 = np.random.randint(i1+1,N)

        off_diag[k] = H[i1,i2]
        diag_diff[k] = H[i1,i1]-H[i2,i2]
    
    return off_diag, diag_diff 



#################### JACOBI functions ####################

@njit
def fill_M(N, H):
    """
    For each row i of H, find the column index of the largest-magnitude off-diagonal element.
    If no off-diagonal exists, set to (i+1)%N to avoid self-pairing.

    Parameters:
    - N: matrix size
    - H: NxN numpy array

    Returns:
    - M: 1D array of length N with the chosen column index for each row
    """
    M = np.zeros(N)
    for i in range(N):
        max_amp = 0
        for j in range(N):
            if j!=i and np.abs(H[i,j]) > max_amp:
                max_amp = np.abs(H[i,j])
                pos = j
        
        M[i] = pos

        if max_amp == 0:
            M[i] = (i+1)%N
    
    return M

@njit
def find_offdiag_M (N, H, M):
    """
    Find the row index `pos` such that |H[pos, M[pos]]| is maximal across rows.

    Parameters:
    - N: matrix size
    - H: NxN array
    - M: index vector mapping rows to their current best column

    Returns:
    - pos: integer row index of the current maximum off-diagonal element
    """
    max_offdiag = 0
    for i in range(N):
        if np.abs(H[i,int(M[i])]) > max_offdiag and i!=int(M[i]):
            max_offdiag = np.abs(H[i,int(M[i])])
            pos = i
    
    return pos

@njit
def update_vecM(N, H, M, pos):
    """
    Update the vector M (best partner indices per row) after a rotation touched row `pos` and its partner.

    Parameters:
    - N: matrix size
    - H: NxN array (modified in place previously)
    - M: current mapping from rows to chosen partner column indices
    - pos: row index of the pivot that was most recently rotated

    Returns:
    - M: updated mapping (modified and returned)
    """
    val = int(M[pos])

    for i in range(N):
        if int(M[i])==val or int(M[i])==pos or i==val or i==pos:
            max_offdiag = 0
            for j in range(N):
                if j != i and np.abs(H[i,j]) >= max_offdiag:
                    max_offdiag = np.abs(H[i,j])
                    posj = j
            M[i] = posj
        
        else:
            if np.abs(H[i,val]) > np.abs(H[i,int(M[i])]) and val!=i:
                M[i] = val
            if np.abs(H[i,pos]) > np.abs(H[i,int(M[i])]) and pos!=i:
                M[i] = pos
    return M


#################### JACOBI ####################

@njit
def get_bin_index(w):
    """
    Convert a positive scalar w into an integer bin index on a logarithmic grid base 1.1.

    Parameters:
    - w: positive scalar (often absolute value of an off-diagonal)

    Returns:
    - integer bin index for bookkeeping iteration counts
    """
    return int(round(np.log(w) / np.log(1/1.1)))


@njit
def jacobi(N, H, iter):
    """
    Simple Jacobi diagonalization loop that repeatedly zeroes the largest selected off-diagonal
    element using a Givens-like rotation chosen from the 2x2 submatrix.

    Parameters:
    - N: matrix size
    - H: NxN real symmetric matrix (numpy array)
    - iter: iteration multiplier; total steps attempted is iter * N

    Returns:
    - niter_vec: vector counting how many iterations happened at each bin index of off-diagonal size
    """
    print("Here")
    
    cutoff = 1e-2

    niter_vec = np.zeros(get_bin_index(cutoff))


    niter_max = iter*N
    

    M = fill_M(N,H)

    

    cumul = 0



    for niter in range(niter_max):

        #print("Here")

        pos = find_offdiag_M(N,H,M)
    

        i1 = pos #np.min([pos,int(M[pos])])
        i2 = int(M[pos]) #np.max([pos,int(M[pos])]) 

        iter_pos = get_bin_index(np.abs(H[i1,i2]))

        if iter_pos < 0:
            iter_pos = 0

        niter_vec[iter_pos] += 1
        
        
        if np.abs(H[i1,i2]) < cutoff:
            print(niter)
            print("Done")
            break
        
        theta = 0.5 *np.arctan(2*H[i1,i2]/(H[i2,i2]-H[i1,i1]))
        if H[i1,i1] == H[i2,i2]:
            theta=np.pi/4
            #print(theta)
        
        
        #angle[ind] = theta

        S = np.zeros((2,2))

        S[0,0] = H[i1,i1]
        S[1,1] = H[i2,i2]
        S[0,1] = H[i1,i2]
        S[1,0] = H[i2,i1]

            
        cumul += (S[0,1])**2
        
        
        c = np.cos(theta)
        s = np.sin(theta)

        H[i1,i1] = c*c*S[0,0] - 2*c*s*S[0,1] + s*s*S[1,1]
        H[i2,i2] = s*s*S[0,0] + 2*c*s*S[0,1] + c*c*S[1,1]

        H[i1,i2] = 0 #(c*c-s*s)*S[0,1] + c*s*(S[0,0]-S[1,1])
        H[i2,i1] = 0 #H[i1,i2]


        for k in range(N):
            if k!= i1 and k!=i2:
                A = H[i1,k]
                B = H[i2,k]

                H[i1,k] = c*A - s*B
                H[i2,k] = s*A + c*B
                H[k,i1] = H[i1,k]
                H[k,i2] = H[i2,k]


        M = update_vecM(N, H, M, pos)

    
    return niter_vec


@numba.njit
def rot_sym(M):
	"""Rotation matrix given 2x2 real symmetric matrix M.
	
	Returns a 2x2 orthogonal rotation that diagonalizes M.
	"""
	Mdiff = (M[1,1]-M[0,0])/2
	Mij = M[0,1]
	tht = np.arctan2(Mij, Mdiff)
	return np.array([[np.cos(tht/2),-np.sin(tht/2)],
					 [np.sin(tht/2),np.cos(tht/2)]])
	
@numba.njit
def rot_sym_eta(M):
	"""Rotation matrix given 2x2 real symmetric matrix M.
	
	Also returns the rotation angle `tht` to allow additional bookkeeping.
	"""
	Mdiff = (M[1,1]-M[0,0])/2
	Mij = M[0,1]
	tht = np.arctan2(Mij, Mdiff)
	return (tht, np.array([[np.cos(tht/2),-np.sin(tht/2)],
					 [np.sin(tht/2),np.cos(tht/2)]]))

@numba.njit
def rot_herm(M):
	"""Rotation matrix given 2x2 complex hermitian matrix M.
	
	Returns a 2x2 unitary that diagonalizes the Hermitian M using a particular phase convention.
	"""
	Mdiff = np.real(M[1,1]-M[0,0])/2
	Mij = M[0,1]
	eitht1 = np.exp(1j*(np.angle(Mij)/2 - np.pi/4))
	tht2 = np.arctan2(np.abs(Mij), Mdiff)/2
	return np.array([[-1j*np.cos(tht2)/eitht1,-np.sin(tht2)*eitht1],
					 [np.sin(tht2)/eitht1,1j*np.cos(tht2)*eitht1]])

@numba.njit
def wt2(M):
	"""Half the squared weight which can be rotated to the diagonal
	in a 2x2 complex matrix M.

	This computes a measure of how much of the 2x2 matrix's norm can be moved
	into the diagonal by a unitary rotation. Used for weighing pivot choices.
	"""
	# Vector of Pauli coefficients giving M.
	a = np.array([(M[0,1]+M[1,0])/2, 1j*(M[0,1]-M[1,0])/2, (M[0,0]-M[1,1])/2])
	
	# Terms appearing in the weight on the diagonal.
	first_term = np.dot(a.real, a.real)**2 + np.dot(a.imag, a.imag)**2 + 2*np.dot(a.real, a.imag)**2
	second_term = np.sum(np.abs(a)**2) * np.abs(np.dot(a,a))
	
	# The weight on the diagonal after rotation, squared. 
	dia = np.sqrt((first_term + second_term)/2)
	# The weight on the off-diagonal that can be removed.
	# Normalized to be |M_01|^2 for Hermitian M.
	return dia - np.abs(a[2])**2

@numba.njit
def maxind(M, row):
	"""Return the column index of the maximum element (in absolute value)
	in the upper triangle of M in the specified row."""
	return row + 1 + np.argmax(np.abs(M[row,row+1:]))


@numba.njit
def update_maxes(touched_inds, M, maxinds, maxels):
    """Update the maximum elements and their recorded indices, given
    a rotation affected all the touched_inds indices.

    Parameters:
    - touched_inds: iterable/list of indices that were rotated (e.g. [i,j])
    - M: full matrix (modified in place prior to this call)
    - maxinds: current recorded partner indices for each row
    - maxels: current recorded maximum element magnitudes for each row

    This updates maxinds and maxels in place.
    """
    # Convert touched_inds to a NumPy array if it’s not already
    touched_inds = np.array(touched_inds)

    #print(touched_inds)

    # Loop over rows which could possibly have been affected.
    for row in numba.prange(min(max(touched_inds)+1, M.shape[0]-1)):
        # If the index got touched by the rotation, we need to search
        # through everything.
        if maxinds[row] in touched_inds or row in touched_inds:
            maxinds[row] = maxind(M, row)
            maxels[row] = np.abs(M[row, maxinds[row]])
        else:
            # If the column index in maxinds is untouched, we just
            # need to compare to the updated elements in this row, which
            # are in touched columns. We also have to check this is still
            # in the upper triangle.
            for ind in touched_inds:
                if maxels[row] < np.abs(M[row, ind]) and ind > row:
                    maxinds[row] = ind
                    maxels[row] = np.abs(M[row, ind])
			

@numba.njit
def jacobi_herm(M, thresh, maxsteps):
    """Perform the Jacobi algorithm on M until the max
    element is less than thresh, or for maxsteps.
    Perform the same rotation thus defined on out.
    M - array to diagonalize
    thresh - matrix element threshold to stop at
    maxsteps - number of steps to stop at

    Returns:
    - niter_vec: list of iteration steps at which pivots were accepted (used for diagnostics)
    - w_vec: list of pivot thresholds used during the run
    """
    niter_vec = []

    w_vec = []

    w_pivot = 10
    w_vec.append(w_pivot)

    niter_vec.append(0)


    # Find the maximum element in each row of the upper triangle,
    # and record their column indices.
    maxinds = np.zeros(M.shape[0]-1, dtype=np.int64)
    maxels = np.zeros(M.shape[0]-1, dtype=np.float64)
    past_maxels = np.zeros((2,maxsteps), dtype=np.float64)
    for row in numba.prange(M.shape[0]-1):
        maxinds[row] = maxind(M, row)
        maxels[row] = np.abs(M[row, maxinds[row]])

    # Locate the row index of the maximum element in the upper triangle.
    i = np.argmax(maxels)
    # The column index is recorded in maxinds.
    j = maxinds[i]

    # Loop over number of steps to take. To actually diagonalize something,
    # replace this with some tolerance.
    step = 1
    while maxels[i] > thresh and step <= maxsteps:

        if maxels[i] < w_pivot:

            niter_vec.append(step)

            w_pivot = w_pivot/1.1

            w_vec.append(w_pivot)

        # Record maximum element.
        past_maxels[0,step-1] = maxels[i]
        # Record corresponding energy difference.
        past_maxels[1,step-1] = (M[i,i] - M[j,j]).real/2

        # Get the matrix which minimizes this element.
        rot = rot_herm(M[i:j+1:j-i,i:j+1:j-i])
        
        ## Update M in place.
        # numba doesn't like this, as the slices below make the arrays non-contiguous.
        # Update rows by left multiplication by rot.
        M[i:j+1:j-i,:] = rot @ M[i:j+1:j-i,:] 
        # Update column by right multiplication by rot.T
        M[:,i:j+1:j-i] = M[:,i:j+1:j-i] @ np.conj(rot.T)
        
        # # Update output in place
        # out[i:j+1:j-i,:] = rot @ out[i:j+1:j-i,:]
        # out[:,i:j+1:j-i] = out[:,i:j+1:j-i] @ np.conj(rot.T)
        
        # Update the maximum elements.
        update_maxes([i,j], M, maxinds, maxels)
        
        # Locate the row index of the maximum element in the upper triangle.
        i = np.argmax(maxels)
        # The column index is recorded in maxinds.
        j = maxinds[i]

        # Increment counter
        step += 1
                
    # Return the decimated elements
    return niter_vec, w_vec


#################### MAIN ####################

L = int( sys.argv[1] )
dis_num = int( sys.argv[2] )
dis_0 = int( sys.argv[3] )

iter = 1000


####################  Parameters  ####################

W = float( sys.argv[4] )


#################### JACOBI ####################

N=2**L

exec(open("matBuilder.py").read())

iter = iter*N

H0 = H0_XXZ(L)


for dis in range(dis_0, dis_0 + dis_num):
    
    #tic = time.perf_counter()
    #H = LNRP_Matrix(N,gamma,1)

    D = sparse.lil_matrix((N,N),dtype=complex)
    for i in range(L):
        D += rng.uniform(-W,W)*sz_list[i]
    

    D = Proj @ D @ Proj.T
    H = H0 + D

    H = np.real(np.array(H.todense()))

    N_H = int(binom(L,L//2))


    niter_vec = jacobi(N_H,H,iter)
    
    #toc = time.perf_counter()
    
    
    filename = "Results_XXZ/niter_XXZ_Jac_L%d_W%.2f_dis%d.txt"%(L,W,dis)
    np.savetxt(filename,niter_vec)
    """
    filename = "Results_XXZ/w_vec_XXZ_Jac_L%d_W%.2f_dis%d.txt"%(L,W,dis)
    np.savetxt(filename,w_vec)
    filename = "Results_XXZ/nres_XXZ_Jac_L%d_W%.2f_dis%d.txt"%(L,W,dis)
    np.savetxt(filename,nres_vec)
    """
