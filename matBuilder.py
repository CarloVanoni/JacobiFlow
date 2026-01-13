#!/usr/bin/env python3
"""
ham_chain_fast.py

Fast, memory-efficient version of ham_chain.py that produces the *same output file*:
    XXZ_OFFDIAG_L{L}.txt

Implements:
    H = sum_i [ S^+_{i+1} S^-_i + S^+_i S^-_{i+1} + Delta * S^z_{i+1} S^z_i ]

Uses spin-1/2 operators (S_z = sigma_z/2) and periodic boundary conditions.
"""

import numpy as np
import scipy.sparse as sp
import sys

def build_xxz_sparse(L, Delta=1.0):
    N = 1 << L  # Hilbert space dimension = 2^L
    rows, cols, data = [], [], []

    for b in range(N):
        diag_val = 0.0
        for i in range(L):
            j = (i + 1) % L

            si = (b >> i) & 1
            sj = (b >> j) & 1

            # Sz Sz contribution
            sz_i = 0.5 if si == 1 else -0.5
            sz_j = 0.5 if sj == 1 else -0.5
            diag_val += Delta * (sz_i * sz_j)

            # Flip-flop terms
            if si != sj:
                bflip = b ^ ((1 << i) | (1 << j))
                rows.append(bflip)
                cols.append(b)
                data.append(1.0)  # amplitude = 1

        # diagonal term
        rows.append(b)
        cols.append(b)
        data.append(diag_val)

    H = sp.csr_matrix(
        (np.array(data, dtype=np.float64),
         (np.array(rows, dtype=np.int32), np.array(cols, dtype=np.int32))),
        shape=(N, N),
        dtype=np.complex128
    )
    return H


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 ham_chain_fast.py L [Delta]")
        sys.exit(1)

    L = int(sys.argv[1])
    Delta = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0

    print(f"Building XXZ Hamiltonian (fast) for L={L}, Delta={Delta}")

    H = build_xxz_sparse(L, Delta)

    print("Matrix built. shape =", H.shape, " nnz =", H.nnz)

    # --- identical output section ---
    rows, cols = H.nonzero()
    vals = np.real(H[rows, cols]).A1  # same as your original code

    filename = f"XXZ_OFFDIAG_L{L}.txt"
    toSave = np.stack((rows, cols, vals)).T
    np.savetxt(filename, toSave, fmt=['%d', '%d', '%d'])
    print("Saved:", filename)
