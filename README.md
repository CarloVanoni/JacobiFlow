# JacobiFlow

This repository contains the code and data used for the paper on the Jacobi flow of resonances. The implementations compute and analyse the flow of matrix elements and resonances for several model classes (XXZ chain, Random Regular Graphs (RRG), and Levy matrices), and include plotting utilities and precomputed data files.

## Repository structure

- jacobi_XXZ.py        - Jacobi-flow implementation for the XXZ spin chain
- jacobi_RRG.py        - Jacobi-flow implementation for Random Regular Graphs (RRG)
- jacobi_Levy.py       - Jacobi-flow implementation for Levy matrices
- decres_XXZ_bin.py    - Binary-decimation / resonance-reduction helper for XXZ
- decres_RRG_bin.py    - Binary-decimation / resonance-reduction helper for RRG
- decres_Levy_bin.py   - Binary-decimation / resonance-reduction helper for Levy
- matBuilder.py        - Utilities to build model matrices used by the main scripts
- plot_theta_XXZ_bin.py
- plot_theta_RRG_bin.py
- plot_theta_Levy_bin.py
- plot_BoE_flow.py     - Plotting utilities for Baker‑von‑Egmond (BoE) flow / observables
- XXZ_OFFDIAG_L*.txt   - Precomputed off-diagonal data for the XXZ model (L=4,6,8,10,12,14,16)

Note: Some data files (especially XXZ_OFFDIAG_L16.txt) are large — take care when cloning or downloading the repository.

## Requirements

- Python 3.8+ recommended
- NumPy
- SciPy
- Matplotlib
- NetworkX (required if building/working with RRG models)
- Optional: Numba (for performance), tqdm (progress bars), seaborn (plot styling)

Install dependencies with pip:

python -m pip install --upgrade pip
python -m pip install numpy scipy matplotlib networkx
# optionally
python -m pip install numba tqdm seaborn


## Basic usage

Each main script is a standalone Python program. Typical usage patterns:

- Inspect available command-line options (if implemented) with:
  python jacobi_XXZ.py --help

- Run a Jacobi-flow simulation (example):
  python jacobi_XXZ.py

- Build matrices from scratch using matBuilder.py (example):
  python matBuilder.py

- Plot results from binary output files:
  python plot_theta_XXZ_bin.py
  python plot_BoE_flow.py

If a script has no CLI, open the header of the script and edit the parameters at the top, or import functions directly from the script in an interactive session or Jupyter notebook.

## Data files

- XXZ_OFFDIAG_L*.txt: text files with precomputed off-diagonal matrix-element data for the XXZ model at different system sizes. They are provided to reproduce figures from the paper without re-running expensive simulations.

If you want to regenerate data you can use the corresponding jacobi_* and decres_* scripts; be aware of their computational cost.

## Reproducing figures

The plotting scripts in the repository generate most of the figures used in the paper. Typical workflow:

1. Generate or obtain the binary/text data using the jacobi_* and decres_* scripts.
2. Use the plotting scripts (plot_theta_*.py and plot_BoE_flow.py) to produce figures.

Look inside each plotting script to find the exact filenames and parameters used to create each figure.

## Performance notes

- Some procedures are computationally heavy (especially for larger L in the XXZ model). If available, enable Numba or run on a machine with more memory/CPU.
- The precomputed XXZ_OFFDIAG_L16.txt file is particularly large; if you only need small sizes, omit downloading L=16 to save space.

## Contributing

Contributions and bug reports are welcome. Please open an issue describing the problem or a feature request, and submit pull requests with clear changes and tests where appropriate.

## License

This repository currently does not include a license file. Please contact the owner (CarloVanoni) to confirm licensing terms before re-using substantial portions of the code.

## Citation

If you use this code in your research, please cite the corresponding paper ("Jacobi flow of resonances"). If you want, provide the DOI or BibTeX entry and I will add a complete citation block here.

## Contact

Owner: https://github.com/CarloVanoni


---

Notes from the assistant: I prepared a README.md describing the project, repository structure, dependencies, usage guidelines, data files and reproduction notes. If you want I can (1) add a full citation/BibTeX entry if you provide the paper reference, (2) include an example notebook demonstrating one full run-to-plot pipeline, or (3) commit the README.md into the repository now.