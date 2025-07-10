Spectral Coarse-Graining Theory
================================

Overview
--------

Spectral Coarse-Graining (SCG) reduces a complex financial network to a smaller representative graph while preserving its essential diffusion dynamics. The method operates on the eigenspectrum of the graph Laplacian.

Graph Laplacian
---------------

Given an adjacency matrix :math:`A` of an interbank network, the normalized Laplacian is:

.. math::

   L = I - D^{-1/2} A D^{-1/2}

where :math:`D` is the degree matrix.

Spectral Decomposition
----------------------

The eigendecomposition :math:`L = V \Lambda V^T` reveals the network's characteristic scales:

- **Algebraic connectivity** :math:`\lambda_2`: measures how well-connected the network is
- **Spectral gap**: the largest jump between consecutive eigenvalues identifies the natural coarse-graining scale
- **Spectral radius** :math:`\rho`: the largest eigenvalue

SCG Risk Score
--------------

The SCG-based systemic risk indicator is defined as:

.. math::

   \text{risk} = 1 - \frac{\lambda_2}{\rho}

This captures both connectivity weakness (low :math:`\lambda_2`) and structural heterogeneity (high :math:`\rho`). The score approaches 1 as the network becomes more fragile.

Coarse-Graining Procedure
--------------------------

1. Compute the Laplacian eigenspectrum
2. Identify the spectral gap at index :math:`k`
3. Group nodes by their dominant eigenvector components (first :math:`k` eigenvectors)
4. Contract grouped nodes into super-nodes
5. Rescale edge weights to preserve diffusion dynamics

Reconstruction Accuracy
-----------------------

The quality of coarse-graining is measured by comparing diffusion on the original and reduced networks:

- **R²**: coefficient of determination between original and reconstructed diffusion
- **RMSE**: root mean square error over time steps

High R² (> 0.9) indicates the coarse-grained network faithfully preserves the contagion dynamics of the full interbank system.
