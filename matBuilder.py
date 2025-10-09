#  ---------------------------------------------------------------------------------------------  #
#
#   This program builds the sparse matrices Sx[i],Sy[i],etc. of a spin chain of length L,
#   acting only on site i.
#
#  ---------------------------------------------------------------------------------------------  #


# program constants are defined already in the main


#  -------------------------------  construct the basic matrices  ------------------------------  #

#sx = np.array( [[0,1],[1,0]], dtype=np.complex_ ) / 2
#sy = np.array( [[0,-1j],[1j, 0]], dtype=np.complex_ ) / 2
sz = np.array( [[1,0],[0,-1]] ) / 2.
sx = np.array( [[0,1],[1,0]] ) / 2.
sy = np.array( [[0,1],[-1,0]] ) / (2.*1j)
sp = np.array( [[0,1],[0,0]] )
sm = np.array( [[0,0],[1,0]] )
id = np.array( [[1,0],[0,1]] )

#sx_list = []; sy_list = []
sz_list = []
sx_list = []
sy_list = []
sp_list = []; sm_list = []
id_list = []

for i in range(L):

    if i==0:
        #full_sx = sx; full_sy = sy
        full_sz = sz
        full_sx = sx
        full_sy = sy
        full_sp = sp; full_sm = sm
        full_id = id
    else:
        #full_sx = sparse.kron(sparse.identity(2**i), sx); full_sy = sparse.kron(sparse.identity(2**i), sy)
        full_sz = sparse.kron(sparse.identity(2**i), sz)
        full_sx = sparse.kron(sparse.identity(2**i), sx)
        full_sy = sparse.kron(sparse.identity(2**i), sy)
        full_sp = sparse.kron(sparse.identity(2**i), sp); full_sm = sparse.kron(sparse.identity(2**i), sm)
        ful_id = sparse.kron(sparse.identity(2**i), id)

    if i!=L-1:
        #full_sx = sparse.kron(full_sx, sparse.identity(2**(L-i-1))); full_sy = sparse.kron(full_sy, sparse.identity(2**(L-i-1)))
        full_sz = sparse.kron(full_sz, sparse.identity(2**(L-i-1)))
        full_sx = sparse.kron(full_sx, sparse.identity(2**(L-i-1)))
        full_sy = sparse.kron(full_sy, sparse.identity(2**(L-i-1)))
        full_sp = sparse.kron(full_sp, sparse.identity(2**(L-i-1))); full_sm = sparse.kron(full_sm, sparse.identity(2**(L-i-1)))
        ful_id = sparse.kron(full_id, sparse.identity(2**(L-i-1)))

    #sx_list.append( sparse.csr_matrix(full_sz) ); sy_list.append( sparse.csr_matrix(full_sz) )  
    sz_list.append( sparse.csr_matrix(full_sz) )  
    sx_list.append( sparse.csr_matrix(full_sx) )  
    sy_list.append( sparse.csr_matrix(full_sy) )  
    sp_list.append( sparse.csr_matrix(full_sp) ); sm_list.append( sparse.csr_matrix(full_sm) )
    id_list.append( sparse.csr_matrix(full_id) )

#del full_sx, full_sy, full_sz
del full_sz, full_sx, full_sy, full_sp, full_sm, full_id


# diagonal of sz matrices
sz_diag_list = [np.array(np.real(sz_list[i].diagonal())) for i in range(L)]
sx_diag_list = [np.array(np.real(sx_list[i].diagonal())) for i in range(L)]
sy_diag_list = [np.array(np.real(sy_list[i].diagonal())) for i in range(L)]
id_diag_list = [np.array(np.real(id_list[i].diagonal())) for i in range(L)]



#  -------------------------------  project on the Sz=0 subspace  ------------------------------  #

# find the vectors with Sz_tot = 0
Sz_tot = 0.
for i in range(L):
    Sz_tot += sz_diag_list[i]
which = np.nonzero(Sz_tot == 0.)[0]

# construct the projector
dimH_red = len(which)
Proj = sparse.lil_matrix((dimH_red,N))
for k in range(dimH_red):
    Proj[k,which[k]] = 1.

Proj = sparse.csr_matrix(Proj)

# project the sz matrices for later convenience
Psz_list = []
for i in range(L):
    #sx_list[i] = Proj @ sx_list[i] @ Proj.T; sy_list[i] = Proj @ sy_list[i] @ Proj.T
    Psz_list.append( Proj @ sz_list[i] @ Proj.T )
    #sp_list[i] = Proj @ sp_list[i] @ Proj.T; sm_list[i] = Proj @ sm_list[i] @ Proj.T


# diagonal of the projected sz matrices
Psz_diag_list = [np.array(np.real(Psz_list[i].diagonal())) for i in range(L)]

"""
# project the sz matrices for later convenience
Pid_list = []
for i in range(L):
    #sx_list[i] = Proj @ sx_list[i] @ Proj.T; sy_list[i] = Proj @ sy_list[i] @ Proj.T
    Pid_list.append( Proj @ id_list[i] @ Proj.T )
    #sp_list[i] = Proj @ sp_list[i] @ Proj.T; sm_list[i] = Proj @ sm_list[i] @ Proj.T


# diagonal of the projected sz matrices
Pid_diag_list = [np.array(np.real(Pid_list[i].diagonal())) for i in range(L)]

"""








