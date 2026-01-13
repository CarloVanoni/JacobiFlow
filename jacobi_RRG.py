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
#from histo_maker import *
import scipy
import time
from numba.typed import List
from scipy.stats import bernoulli, lognorm


rng = default_rng()

@njit
def compute_r(data):
    """
    Compute local ratios of sorted neighboring differences.

    This function sorts the input `data`, takes a window around the
    midpoint (N//2-25:N//2+25), computes successive differences and
    returns the elementwise ratio min(diff[i], diff[i+1]) / max(diff[i], diff[i+1])
    for those differences. This is useful for local spacing-ratio analyses.

    Note: `N` is referenced but not passed; it must be available in the
    calling scope (this mirrors the original code behaviour).
    """
    data_n = np.sort(data)
    diff = np.diff(data_n[N//2-25:N//2+25])
    return np.minimum(diff[:-1], diff[1:]) / np.maximum(diff[:-1], diff[1:])

@jit(nopython=True)
def rand_choice_nb(arr, prob):
    """
    Sample two indices from `arr` according to probability vector `prob`.

    Args:
        arr (1D array): values to sample from.
        prob (1D array): probabilities for each value (should sum to 1).

    Returns:
        tuple: two samples from `arr`, drawn independently with replacement.
    """
    cumsum_prob = np.cumsum(prob)  # pre-calculate cumulative sum
    #rand_nums = np.random.random(2)  # generate two random numbers once
    idx1 = np.searchsorted(cumsum_prob, np.random.random(), side="right")
    idx2 = np.searchsorted(cumsum_prob, np.random.random(), side="right")
    return arr[idx1], arr[idx2]


@numba.jit(nopython=True)
def get_bin_edges(a, bins):
    """
    Compute equally spaced bin edges between min and max of array `a`.

    Args:
        a (array): input values.
        bins (int): number of bins.

    Returns:
        array: (bins+1,) array of bin edges from a.min() to a.max().
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
    Put a value x into a bin index based on `bin_edges` (uniform bins assumed).

    Returns the integer bin index, or None if x outside range.
    """
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
    Fast histogram implementation compatible with numba.

    Returns:
        hist (array): counts per bin
        bin_edges (array): edges used
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
    Spectral weighting function f_V used to scale off-diagonal elements.

    It returns the symmetric sum of two Gaussians centered at +/- omega_0.
    """
    return 1/(2*np.sqrt(2*np.pi*sigma_omega**2)) * ( np.exp(-((omega-omega_0)**2)/(2*sigma_omega**2) ) + np.exp(-((omega+omega_0)**2)/(2*sigma_omega**2) ))

#@numba.jit(nopython=True)
def off_diagonal_elements(matrix, n):
    """
    Randomly select `n` off-diagonal elements and corresponding diagonal differences.

    Args:
        matrix (2D array): square matrix to sample from.
        n (int): number of off-diagonal elements to select.

    Returns:
        off_diagonal (array): selected upper-triangle off-diagonal values
        diagonal_diff (array): corresponding diagonal differences (H[ii,ii]-H[jj,jj])
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
    Generate `n` samples from a symmetric power-law distribution with exponent related to theta.

    This implementation uses an inverse-transform-like sampling over a truncated range [x_min, x_max]
    and randomly assigns signs to obtain a symmetric distribution.

    Args:
        n (int): number of samples
        theta (float): parameter that determines the power-law exponent (alpha = 2 - theta)
        x_min, x_max (float): truncation range for the magnitude

    Returns:
        ndarray: length-n array of samples
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
    Place a 1D list of `values` into the strictly upper-triangular part of a (size x size) matrix.

    Values are filled row-wise into positions with i < j.
    """
    upper = np.zeros((size, size))
    upper[np.triu_indices(size, 1)] = values
    return(upper)


@numba.jit(nopython=True)
def build_FGR_H(N,sigma_E,omega_0,sigma_omega,J):
    """
    Build an ensemble Hamiltonian H for the FGR (Fermi Golden Rule) model.

    Diagonal entries are uniform in [-sigma_E/2, sigma_E/2]. Off-diagonals are Gaussian
    scaled by the spectral function f_V and normalized by sqrt(N/sigma_E).

    Returns:
        H (N x N array), off_val_tot (sum of off-diagonal squared values times 2)
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
    Variant of build_FGR_H where off-diagonals are drawn from a uniform box distribution
    (instead of Gaussian) and scaled similarly.
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
    Build a symmetric matrix whose off-diagonal entries follow a power-law distribution.

    Diagonal entries are uniform in [-2,2].
    Returns:
        H (N x N array), off_val_tot (sum of off-diagonal squared values times 2)
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
    Construct a GOE-like random matrix with an additional set of large entries.

    Args:
        N (int): matrix size
        t (int): number parameter controlling how many extra large entries are added

    Returns:
        H (N x N array), off_val_tot (sum of off-diagonal squared values)
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
    Generate a power-law banded random matrix (PLBRM) and return the list of off-diagonal values and the matrix.

    Args:
        N (int): size
        a,b (float): parameters controlling decay
        W (float): diagonal variance
    Returns:
        init (list): list of off-diagonal entries appended in construction order
        H (N x N array): symmetric matrix
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


# Create upper triangular matrix
def create_upper_matrix(values, size):
    """
    Duplicate of create_upper_matrix above — keep for compatibility with code paths that expect this function.
    """
    upper = np.zeros((size, size))
    upper[np.triu_indices(size, 1)] = values
    return(upper)


def LNRP_Matrix(n,gamma,p):
    """
    Construct a matrix with log-normal-like random off-diagonal elements scaled with n^{-gamma/2}.

    Diagonal elements are uniform in [-1,1].

    Args:
        n (int): matrix size
        gamma (float): exponent controlling scaling
        p (float): additional parameter used in lognormal variance
    Returns:
        H (n x n array)
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
    Generate n random numbers distributed according to a truncated symmetric power-law:
      p(x) ~ x^{-(1+mu)} for |x| > N^{-gamma/mu}

    Uses inverse-transform sampling for absolute values and random signs.

    Args:
        mu (float): exponent parameter
        N, gamma: parameters controlling cutoff
        n (int): number of samples
    Returns:
        numpy.ndarray of length n
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
    Build a symmetric matrix with heavy-tailed (Levy/power-law) off-diagonal elements.

    Diagonal entries are uniform in [-W/2, W/2].
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
    Construct a hierarchical (kron-structured) unitary-like matrix built from exponentials of small random Hermitian blocks.

    This function builds progressively larger matrices using Kronecker products and applies small exponentials of random
    log-normal-weighted Hermitian matrices. It is specialized for a particular multiscale construction used in the project.

    Args:
        L (int): maximum level
        eps, x, W: scaling parameters used in the block construction
    Returns:
        M (dense matrix): resulting matrix
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

"""
def build_H1(L,J,W,tau):
    Z=np.array([[1,0],[0,-1]])
    X=np.array([[0,1],[1,0]])
    Y=np.array([[0,1],[-1,0]])/ (1j)
    Id=np.array([[1,0],[0,1]])
    H1=0
    for i in range(L-1):
        H1 += reduce(lambda x, y: sps.kron(x,y,format='csc'), [Id]*i+[(J-0.1)*Y,Y]+[Id]*(L-2-i)).todense()
        H1 += reduce(lambda x, y: sps.kron(x,y,format='csc'), [Id]*i+[(J+0.1)*X,X]+[Id]*(L-2-i)).todense()
    
    for i in range(L):
        H1 += reduce(lambda x, y: sps.kron(x,y,format='csc'), [Id]*i+[2*X]+[Id]*(L-1-i)).todense()
        
    H1 += reduce(lambda x, y: sps.kron(x, y, format='csc'), [(J+0.1)*X] + [Id]*(L-2) + [X]).todense()
    H1 += reduce(lambda x, y: sps.kron(x, y, format='csc'), [(J-0.1)*Y] + [Id]*(L-2) + [Y]).todense()
      
    U = expm(-1j*H1*tau)
    Udag = expm(1j*H1*tau)

    XL2 = reduce(lambda x, y: sps.kron(x,y,format='csc'), [Id]*(L//2)+[X]+[Id]*(L//2-1))
    XL2 = XL2.todense()
    M=np.matmul(np.matmul(U,XL2),Udag)

    for i in range(L):
        M += reduce(lambda x, y: sps.kron(x,y,format='csc'), [Id]*i+[(2*np.random.rand()-1)*W*Z]+[Id]*(L-1-i)).todense()
    for i in range(L-1):
        M += reduce(lambda x, y: sps.kron(x,y,format='csc'), [Id]*i+[Z,Z]+[Id]*(L-2-i)).todense()
    M += reduce(lambda x, y: sps.kron(x, y, format='csc'), [Z] + [Id]*(L-2) + [Z]).todense()

    return M
"""

def H_RRG(N, W):
    """
    Build a random-regular-graph adjacency matrix of degree 3 (3-regular graph) and add random on-site disorder.

    Diagonal disorder is uniform in [-W/2, W/2].

    Returns:
        H (N x N dense array)
    """
    G = nx.random_regular_graph(3,N)

    H = nx.adjacency_matrix(G, dtype=float)

    H = H.todense()

    for i in range(N):
        H[i,i] = np.random.uniform(-W/2,W/2)

    return H


def fill_dist_RRG(N):
    """
    Compute all-pairs shortest path distances on a random regular graph of size N (degree 3).

    Returns:
        dist (N x N array) of integer graph distances
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
    Load (or construct) Bethe-lattice-like adjacency from a file and compute pairwise distances.

    The procedure reads adjacency data from 'Adjacency/AdM_L{L}.txt' where L is deduced from N.
    Returns:
        dist (N x N array)
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
    Fill a matrix for the RRG-Jacobi model, where off-diagonal couplings decay exponentially with graph distance.

    Off-diagonals generated as w0 * exp(-dist/z) * Gaussian / sqrt(2**dist).
    Diagonal disorder uniform in [-W/2, W/2].

    Returns:
        H (N x N array), off_val_tot (sum of off-diagonal squared elements)
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
    Similar to fill_RRG_Jac but also returns the list of generated off-diagonal values (init).

    Returns:
        init (list of off-diagonals), H (matrix), off_val_tot (sum of squares)
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
    Compute mean absolute spacing between sorted off-diagonal values (used as a typical spacing measure).
    """
    return np.mean(np.abs(np.diff(np.sort(off_diag))))


@numba.jit(nopython=True)
def off_diag_SRA_FGR(N,n,sigma_E,omega_0,sigma_omega,J):
    """
    Generate `n` synthetic off-diagonal couplings and diagonal differences for SRA/FGR models.

    Returns:
        off_diag, diag_diff, dh (mean spacing), off_val_tot (sum squares), level_spacing (mean diag spacing)
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
    Similar to off_diag_SRA_FGR but uses uniform box distribution for some variables.
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
    Construct synthetic off-diagonals using the power-law distribution helper.
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
    Sample `n` off-diagonal elements and their diagonal differences from matrix H randomly.

    Args:
        H (array): NxN matrix
        N (int): matrix dimension
        n (int): number of samples

    Returns:
        off_diag (array), diag_diff (array)
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
    For each row i, find index of the largest off-diagonal element in that row.
    If a row has all zeros, set its partner to (i+1)%N.
    Returns:
        M (array of length N) where M[i] is the column index of that max element.
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
    Find the row index whose selected partner M[row] yields the globally largest off-diagonal element.
    Returns:
        pos (int): the row index where |H[row, M[row]]| is maximized
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
    Update the vector M of best partners after a rotation affecting index `pos`.

    For any row i that referenced the rotated indices or is those indices, recompute their best partner.
    Otherwise check if the rotation introduced a new larger coupling to pos or M[pos].
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
    Map a weight w to a logarithmic bin index used for bookkeeping iterations.
    Bins are based on powers of 1/1.1 (approximately 10% steps).
    """
    return int(round(np.log(w / 1) / np.log(1/1.1)))



@njit
def jacobi(N, H, iter):
    """
    Perform a Jacobi-like diagonalization using the heuristic partner selection M,
    record iteration counts per bin until a cutoff is reached or max iterations.

    Returns:
        niter_vec: array counting operations per bin
    """
    niter_max = iter*N
    

    M = fill_M(N,H)

    Umat = np.zeros((N,N))
    for i in range(N):
        Umat[i,i] = 1
    
    Gmat = np.zeros((N,N))
    for i in range(N):
        Gmat[i,i] = 1

    cutoff = 1e-3
    
    niter_vec = np.zeros(get_bin_index(cutoff))

    cumul = 0


    for niter in range(niter_max):

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


        #print(H[i1,i2])

        
        for k in range(N): 

            Ui = Umat[i1,k]
            Uj = Umat[i2,k]

            Umat[i1,k] = s*Uj + c*Ui
            Umat[i2,k] = c*Uj - s*Ui

            Gi = Gmat[k,i1]
            Gj = Gmat[k,i2]

            Gmat[k,i1] = - s*Gj + c*Gi
            Gmat[k,i2] = c*Gj + s*Gi

            if k!= i1 and k!=i2:
                A = H[i1,k]
                B = H[i2,k]

                H[i1,k] = c*A - s*B
                H[i2,k] = s*A + c*B
                H[k,i1] = H[i1,k]
                H[k,i2] = H[i2,k]


        M = update_vecM(N, H, M, pos)

    
    return niter_vec



@njit
def jacobi_bin(N, H, iter):
    """
    Same as `jacobi` but without printing progress; returns iteration histogram per bin.
    """
    niter_max = iter*N
    

    M = fill_M(N,H)

    Umat = np.zeros((N,N))
    for i in range(N):
        Umat[i,i] = 1
    
    Gmat = np.zeros((N,N))
    for i in range(N):
        Gmat[i,i] = 1

    cutoff = 1e-3
    
    niter_vec = np.zeros(get_bin_index(cutoff))

    cumul = 0


    for niter in range(niter_max):

        pos = find_offdiag_M(N,H,M)
    

        i1 = pos #np.min([pos,int(M[pos])])
        i2 = int(M[pos]) #np.max([pos,int(M[pos])]) 

        iter_pos = get_bin_index(np.abs(H[i1,i2]))

        if iter_pos < 0:
            iter_pos = 0

        niter_vec[iter_pos] += 1

        
        if np.abs(H[i1,i2]) < cutoff:
            break
        
        theta = 0.5 *np.arctan(2*H[i1,i2]/(H[i2,i2]-H[i1,i1]))

        if H[i1,i1] == H[i2,i2]:
            theta=np.pi/4
        
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


        #print(H[i1,i2])

        
        for k in range(N): 

            Ui = Umat[i1,k]
            Uj = Umat[i2,k]

            Umat[i1,k] = s*Uj + c*Ui
            Umat[i2,k] = c*Uj - s*Ui

            Gi = Gmat[k,i1]
            Gj = Gmat[k,i2]

            Gmat[k,i1] = - s*Gj + c*Gi
            Gmat[k,i2] = c*Gj + s*Gi

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
	"""Rotation matrix given 2x2 real symmetric matrix M."""
	Mdiff = (M[1,1]-M[0,0])/2
	Mij = M[0,1]
	tht = np.arctan2(Mij, Mdiff)
	return np.array([[np.cos(tht/2),-np.sin(tht/2)],
					 [np.sin(tht/2),np.cos(tht/2)]])
	
@numba.njit
def rot_sym_eta(M):
	"""Rotation matrix given 2x2 real symmetric matrix M."""
	Mdiff = (M[1,1]-M[0,0])/2
	Mij = M[0,1]
	tht = np.arctan2(Mij, Mdiff)
	return (tht, np.array([[np.cos(tht/2),-np.sin(tht/2)],
					 [np.sin(tht/2),np.cos(tht/2)]]))

@numba.njit
def rot_herm(M):
	"""Rotation matrix given 2x2 complex hermitian matrix M."""
	Mdiff = np.real(M[1,1]-M[0,0])/2
	Mij = M[0,1]
	eitht1 = np.exp(1j*(np.angle(Mij)/2 - np.pi/4))
	tht2 = np.arctan2(np.abs(Mij), Mdiff)/2
	return np.array([[-1j*np.cos(tht2)/eitht1,-np.sin(tht2)*eitht1],
					 [np.sin(tht2)/eitht1,1j*np.cos(tht2)*eitht1]])

@numba.njit
def wt2(M):
	"""Half the squared weight which can be rotated to the diagonal
	in a 2x2 complex matrix M."""
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
    a rotation affected all the touched_inds indices."""
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
    out - output to similarly rotate. CURRENTLY REMOVED.
    thresh - matrix element threshold to stop at
    maxsteps - number of steps to stop at'"""

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

iter = iter*N



for dis in range(dis_0, dis_0 + dis_num):
    
    #tic = time.perf_counter()
    #H = LNRP_Matrix(N,gamma,1)

    H = H_RRG(N,W)

    H = np.array(H)

    niter_vec = jacobi_bin(N,H,iter)
    
    #toc = time.perf_counter()
    
    
    filename = "Results_RRG_bin/niter_RRG_Jac_L%d_W%.2f_dis%d.txt"%(L,W,dis)
    np.savetxt(filename,niter_vec)
    
