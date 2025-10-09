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
#from histo_maker import *
import scipy
import time
from numba.typed import List
from scipy.stats import bernoulli, lognorm


rng = default_rng()

@njit
def compute_r(data):
    data_n = np.sort(data)
    diff = np.diff(data_n[N//2-25:N//2+25])
    return np.minimum(diff[:-1], diff[1:]) / np.maximum(diff[:-1], diff[1:])

@jit(nopython=True)
def rand_choice_nb(arr, prob):
    """
    :param arr: A 1D numpy array of values to sample from.
    :param prob: A 1D numpy array of probabilities for the given samples.
    :return: Two random samples from the given array with the given probabilities.
    """
    cumsum_prob = np.cumsum(prob)  # pre-calculate cumulative sum
    #rand_nums = np.random.random(2)  # generate two random numbers once
    idx1 = np.searchsorted(cumsum_prob, np.random.random(), side="right")
    idx2 = np.searchsorted(cumsum_prob, np.random.random(), side="right")
    return arr[idx1], arr[idx2]


@numba.jit(nopython=True)
def get_bin_edges(a, bins):
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
    hist = np.zeros((bins,), dtype=np.intp)
    bin_edges = get_bin_edges(a, bins)

    for x in a.flat:
        bin = compute_bin(x, bin_edges)
        if bin is not None:
            hist[int(bin)] += 1

    return hist, bin_edges


@numba.jit(nopython=True)
def f_V(omega,omega_0,sigma_omega):
    return 1/(2*np.sqrt(2*np.pi*sigma_omega**2)) * ( np.exp(-((omega-omega_0)**2)/(2*sigma_omega**2) ) + np.exp(-((omega+omega_0)**2)/(2*sigma_omega**2) ))

#@numba.jit(nopython=True)
def off_diagonal_elements(matrix, n):
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
    number = np.array([])
    alpha = 2-theta
    for i in range(n):
        norm = x_max**(1-alpha) - x_min**(1-alpha)
        u_val = random.uniform(0, 1)
        number = np.append(number,random.choice((-1, 1)) * (u_val*norm + x_min**(1-alpha))**(1/(1-alpha)))

    return number


def create_upper_matrix(values, size):
    upper = np.zeros((size, size))
    upper[np.triu_indices(size, 1)] = values
    return(upper)


@numba.jit(nopython=True)
def build_FGR_H(N,sigma_E,omega_0,sigma_omega,J):
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
    r = powerLawDistribution(N*(N-1)//2,theta,0.001,1)
    off_val_tot = 2*np.sum(r**2)

    # Fills the upper triangular matrix
    c = create_upper_matrix(r, N)

    #Diagonal
    D = np.random.uniform(-2,2,N)
    H = c + c.T + np.diag(D)

    return H, off_val_tot


def build_GOE_H(N,t):
    H = rng.standard_normal((N,N))/np.sqrt(2*N)
    big_inds = rng.integers(N, size=N*t)
    H[np.repeat(np.arange(N),t),big_inds] += rng.standard_normal(N*t)/np.sqrt(2*t)
    H += H.T

    off_val_tot = np.sum(np.square(H)) - np.sum(np.square(np.diag(H)))

    return H, off_val_tot


@numba.jit(nopython=True)
def plbrm(N,a,b,W):
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
    upper = np.zeros((size, size))
    upper[np.triu_indices(size, 1)] = values
    return(upper)


def LNRP_Matrix(n,gamma,p):

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
    mu (float): Shape parameter of the power-law distribution.
    N (float): A normalization or scale factor.
    gamma (float): Controls the threshold for the truncation.
    n (int): Number of random values to generate.
    
    Returns:
    numpy.ndarray: Array of n random numbers following the desired distribution.
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

    # Extract LN off diag elements
    r = random_power_law(mu, n, gamma, n*(n-1)//2)

    # Fills the upper triangular matrix
    c = create_upper_matrix(r, n)

    #Diagonal
    D = rng.uniform(-W/2,W/2,size=n)

    # Full matrix
    H = c + c.T + np.diag(D)

    return H




def fill_dist_RRG(N):
    G = nx.random_regular_graph(3,N)

    dist = np.zeros((N,N))

    for i in range(N):
        for j in range(i+1,N):
            dist[i,j] = nx.shortest_path_length(G, i, j)
            dist[j,i] = dist[i,j]

    return dist


def fill_Bethe(N):
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
    return np.mean(np.abs(np.diff(np.sort(off_diag))))


@numba.jit(nopython=True)
def off_diag_SRA_FGR(N,n,sigma_E,omega_0,sigma_omega,J):
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
    max_offdiag = 0
    for i in range(N):
        if np.abs(H[i,int(M[i])]) > max_offdiag and i!=int(M[i]):
            max_offdiag = np.abs(H[i,int(M[i])])
            pos = i
    
    return pos

@njit
def update_vecM(N, H, M, pos):
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
def jacobi(N, H, iter):

    up_lim = np.max(np.diag(H))
    low_lim = np.min(np.diag(H))

    nres = 0


    niter_max = iter*N
    

    M = fill_M(N,H)

    Umat = np.zeros((N,N))
    for i in range(N):
        Umat[i,i] = 1
    
    Gmat = np.zeros((N,N))
    for i in range(N):
        Gmat[i,i] = 1


    niter_vec = []

    nres_vec = []

    ipr_vec = []
    

    cumul = 0


    diag = np.diag(H)
    diag = np.sort(diag)

    #off_w = []

    level_spacing = np.abs(np.mean(np.diff(diag)))

    #Log-spaced w values, ranging from 1 to level spacing
    w_vec = []

    w_pivot = 10
    w_vec.append(w_pivot)

    niter_vec.append(0)
    nres_vec.append(0)


    for niter in range(niter_max):

        pos = find_offdiag_M(N,H,M)
    

        i1 = pos #np.min([pos,int(M[pos])])
        i2 = int(M[pos]) #np.max([pos,int(M[pos])]) 

        

        if np.abs(H[i1,i2]) < w_pivot:

            niter_vec.append(niter)

            nres_vec.append(nres)

            ipr_w = []

            for k in range(N):
                ipr_w.append(sum(Umat[:,k]**4))

            ipr_vec.append(sum(ipr_w)/N)

            w_pivot = w_pivot/1.1

            w_vec.append(w_pivot)

            diag = np.diag(H)
            diag = np.sort(diag)

            level_spacing = np.min(np.abs(np.diff(diag)))
        
        if np.abs(H[i1,i2]) < level_spacing:
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

        
        #omega.append(H[i2,i2]-H[i1,i1])
        if np.abs(theta)>np.pi/8:
            nres += 1
            #res_off = np.append(res_off,np.abs(S[0,1]))
            
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

    
    return niter_vec, w_vec, nres_vec, ipr_vec




@njit
def get_bin_index(w):
    return int(round(np.log(w / 10) / np.log(1/1.1)))



@njit
def jacobi_bin(N, H, iter):


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




#################### MAIN ####################

N = int( sys.argv[1] )
dis_num = int( sys.argv[2] )
dis_0 = int( sys.argv[3] )


############  Parameters  ##############

gamma = float( sys.argv[4] )

mu = float( sys.argv[5] )

W = float( sys.argv[6] )


#################### JACOBI ####################


iter = 1000

for dis in range(dis_0, dis_0 + dis_num):
    

    H = Levy_Matrix(N,gamma,mu,W)
   

    niter_vec = jacobi_bin(N,H,iter)
    
    filename = "Results_Levy_bin/niter_Levy_Jac_N%d_gamma%.2f_mu%.2f_W%.2f_dis%d.txt"%(N,gamma,mu,W,dis)
    np.savetxt(filename,niter_vec)
