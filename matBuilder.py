#!/usr/bin/env python3
"""
matBuilder.py

Builds the XXZ spin-1/2 chain Hamiltonian with periodic boundary conditions
and writes the nonzero matrix elements to a text file named
    XXZ_OFFDIAG_L{L}.txt

Hamiltonian implemented (conventional spin-1/2 XXZ form):
    H = sum_i [ S^+_{i+1} S^-_i + S^+_i S^-_{i+1} + Delta * S^z_{i+1} S^z_i ]

Notes:
- We represent each basis state as an integer b in [0, 2^L), whose binary
  expansion encodes the spin configuration on L sites:
    bit i of b = 1  -> spin up at site i  (S^z = +1/2)
    bit i of b = 0  -> spin down at site i (S^z = -1/2)
- Periodic boundary conditions: site j = (i+1) % L
- The matrix is constructed in coordinate form (row, col, data lists) and
  converted to a CSR sparse matrix for compact storage.
- The output file contains rows: (row_index, col_index, value) for every
  nonzero matrix element; value is written as a floating point number.
"""

import numpy as np
import scipy.sparse as sp
import sys

def build_xxz_sparse(L, Delta=1.0):
    """
    Build the sparse XXZ Hamiltonian for a spin-1/2 chain of length L.

    Parameters
    ----------
    L : int
        Number of spin sites in the chain.
    Delta : float, optional
        Anisotropy parameter for the Sz Sz interaction (default 1.0).

    Returns
    -------
    H : scipy.sparse.csr_matrix
        The Hamiltonian as a CSR sparse matrix of shape (2**L, 2**L).
        The matrix dtype is complex128 for compatibility with complex
        routines; all entries produced here are real.
    """
    # Hilbert space dimension = 2^L
    N = 1 << L

    # We'll accumulate the matrix in coordinate (COO-like) format:
    rows = []  # destination row indices
    cols = []  # source column indices
    data = []  # matrix values

    # Loop over all computational-basis states b (integer from 0 to 2^L - 1).
    # Each b represents a bitstring of spins.
    for b in range(N):
        # diagonal contribution for this basis state
        diag_val = 0.0

        # Loop over all bonds (i, j=i+1) with periodic BCs
        for i in range(L):
            j = (i + 1) % L

            # Extract the bits (0 or 1) at site i and j.
            # We use the convention: bit == 1 -> spin up (+1/2), bit == 0 -> spin down (-1/2)
            si = (b >> i) & 1
            sj = (b >> j) & 1

            # Sz Sz diagonal contribution for this bond:
            # S^z = +/- 1/2 so product is +/- 1/4; multiply by Delta
            sz_i = 0.5 if si == 1 else -0.5
            sz_j = 0.5 if sj == 1 else -0.5
            diag_val += Delta * (sz_i * sz_j)

            # Flip-flop (off-diagonal) terms:
            # S^+_{i+1} S^-_i + S^+_i S^-_{i+1} acts non-trivially only when
            # the neighboring spins are opposite (one up, one down).
            # In the computational basis that means si != sj. Flipping both bits
            # at positions i and j gives the target basis index bflip.
            if si != sj:
                # Toggle bits i and j using XOR with mask that has 1s at i and j
                bflip = b ^ ((1 << i) | (1 << j))

                # Add the matrix element <bflip|H|b> = 1 (amplitude for flip-flop)
                # Note: when iterating over all b, the reverse element <b|H|bflip>
                # will be added when b == bflip, so the Hamiltonian ends up Hermitian.
                rows.append(bflip)
                cols.append(b)
                data.append(1.0)  # amplitude of the flip-flop term

        # Add the diagonal matrix element for |b>
        rows.append(b)
        cols.append(b)
        data.append(diag_val)

    # Convert lists to numpy arrays with appropriate dtypes for sparse construction.
    # Use int32 index arrays to reduce memory when L is modest; float64 data is sufficient.
    H = sp.csr_matrix(
        (np.array(data, dtype=np.float64),
         (np.array(rows, dtype=np.int32), np.array(cols, dtype=np.int32))),
        shape=(N, N),
        dtype=np.complex128  # choose complex dtype for general compatibility
    )
    return H


if __name__ == "__main__":
    # Simple CLI: python3 matBuilder.py L [Delta]
    if len(sys.argv) < 2:
        print("Usage: python3 matBuilder.py L [Delta]")
        sys.exit(1)

    L = int(sys.argv[1])
    Delta = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0

    print(f"Building XXZ Hamiltonian for L={L}, Delta={Delta}")

    H = build_xxz_sparse(L, Delta)

    print("Matrix built. shape =", H.shape, " nnz =", H.nnz)

    # --- produce identical-format output: list of nonzero entries ---
    # We extract the nonzero coordinates and values and write them to a text file.
    # Note: H.nonzero() returns two arrays (rows, cols) of indices.
    rows, cols = H.nonzero()

    # H[rows, cols] returns a sparse matrix with those entries; convert to dense 1D array.
    # We take the real part because all entries here are real by construction.
    vals = np.real(H[rows, cols]).A1

    filename = f"XXZ_OFFDIAG_L{L}.txt"
    # Use floating-point format for the values. Previously using '%d' would truncate floats.
    toSave = np.stack((rows, cols, vals)).T
    np.savetxt(filename, toSave, fmt=['%d', '%d', '%.6f'])
    print("Saved:", filename)
