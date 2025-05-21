"""
Tests for the spectral analysis module.

This module contains tests for spectral analysis functions.
"""

import unittest
import numpy as np
import networkx as nx

from scr_financial.network.spectral import (
    compute_laplacian,
    eigendecomposition,
    find_spectral_gap,
    compute_diffusion_modes,
    analyze_spectral_properties,
    compute_spectral_embedding,
    compute_diffusion_distance
)


class TestSpectralAnalysis(unittest.TestCase):
    """Test cases for spectral analysis functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a simple test graph
        self.G = nx.DiGraph()
        
        # Add nodes
        self.G.add_nodes_from(['A', 'B', 'C', 'D', 'E'])
        
        # Add edges
        self.G.add_edge('A', 'B', weight=1.0)
        self.G.add_edge('B', 'C', weight=0.8)
        self.G.add_edge('C', 'D', weight=0.6)
        self.G.add_edge('D', 'E', weight=0.4)
        self.G.add_edge('E', 'A', weight=0.2)
        self.G.add_edge('A', 'C', weight=0.5)
        self.G.add_edge('B', 'D', weight=0.3)
        self.G.add_edge('C', 'E', weight=0.7)
        self.G.add_edge('D', 'A', weight=0.9)
        self.G.add_edge('E', 'B', weight=0.1)
        
        # Get adjacency matrix
        self.adjacency_matrix = nx.to_numpy_array(self.G, weight='weight')
    
    def test_compute_laplacian(self):
        """Test computation of Laplacian matrix."""
        # Compute normalized Laplacian
        laplacian = compute_laplacian(self.adjacency_matrix, normalized=True)
        
        self.assertIsInstance(laplacian, np.ndarray)
        self.assertEqual(laplacian.shape, self.adjacency_matrix.shape)
        
        # Check properties of normalized Laplacian
        eigenvalues = np.linalg.eigvals(laplacian)
        
        # Eigenvalues should be non-negative
        self.assertTrue(np.all(np.real(eigenvalues) >= -1e-10))
        
        # Compute unnormalized Laplacian
        laplacian_unnorm = compute_laplacian(self.adjacency_matrix, normalized=False)
        
        self.assertIsInstance(laplacian_unnorm, np.ndarray)
        self.assertEqual(laplacian_unnorm.shape, self.adjacency_matrix.shape)
    
    def test_eigendecomposition(self):
        """Test eigendecomposition."""
        # Compute Laplacian
        laplacian = compute_laplacian(self.adjacency_matrix)
        
        # Perform eigendecomposition
        eigenvalues, eigenvectors = eigendecomposition(laplacian)
        
        self.assertIsInstance(eigenvalues, np.ndarray)
        self.assertIsInstance(eigenvectors, np.ndarray)
        self.assertEqual(len(eigenvalues), laplacian.shape[0])
        self.assertEqual(eigenvectors.shape, laplacian.shape)
        
        # Check that eigenvalues are sorted
        self.assertTrue(np.all(np.diff(eigenvalues) >= -1e-10))
        
        # Check that eigenvectors are orthogonal
        for i in range(eigenvectors.shape[1]):
            for j in range(i + 1, eigenvectors.shape[1]):
                dot_product = np.abs(np.dot(eigenvectors[:, i], eigenvectors[:, j]))
                self.assertAlmostEqual(dot_product, 0.0, places=10)
    
    def test_find_spectral_gap(self):
        """Test finding spectral gap."""
        # Generate some eigenvalues with a clear gap
        eigenvalues = np.array([0.0, 0.1, 0.2, 0.8, 0.9])
        
        # Find spectral gap
        k, gap = find_spectral_gap(eigenvalues)
        
        self.assertEqual(k, 2)  # Gap is between eigenvalues 2 and 3
        self.assertAlmostEqual(gap, 0.6)  # Gap size is 0.8 - 0.2 = 0.6
    
    def test_compute_diffusion_modes(self):
        """Test computing diffusion modes."""
        # Compute Laplacian
        laplacian = compute_laplacian(self.adjacency_matrix)
        
        # Compute diffusion modes
        k = 2
        eigenvalues, eigenvectors = compute_diffusion_modes(laplacian, k)
        
        self.assertIsInstance(eigenvalues, np.ndarray)
        self.assertIsInstance(eigenvectors, np.ndarray)
        self.assertEqual(len(eigenvalues), k + 1)  # k + 1 modes (including constant mode)
        self.assertEqual(eigenvectors.shape[1], k + 1)
    
    def test_analyze_spectral_properties(self):
        """Test analyzing spectral properties."""
        # Compute Laplacian and perform eigendecomposition
        laplacian = compute_laplacian(self.adjacency_matrix)
        eigenvalues, eigenvectors = eigendecomposition(laplacian)
        
        # Analyze spectral properties
        properties = analyze_spectral_properties(eigenvalues, eigenvectors)
        
        self.assertIsInstance(properties, dict)
        self.assertIn('spectral_gap_index', properties)
        self.assertIn('spectral_gap_size', properties)
        self.assertIn('algebraic_connectivity', properties)
        self.assertIn('spectral_radius', properties)
        self.assertIn('participation_ratios', properties)
        self.assertIn('localization', properties)
    
    def test_compute_spectral_embedding(self):
        """Test computing spectral embedding."""
        # Compute Laplacian
        laplacian = compute_laplacian(self.adjacency_matrix)
        
        # Compute spectral embedding
        dim = 2
        embedding = compute_spectral_embedding(laplacian, dim)
        
        self.assertIsInstance(embedding, np.ndarray)
        self.assertEqual(embedding.shape, (laplacian.shape[0], dim))
    
    def test_compute_diffusion_distance(self):
        """Test computing diffusion distance."""
        # Compute Laplacian
        laplacian = compute_laplacian(self.adjacency_matrix)
        
        # Compute diffusion distance
        t = 1.0
        distance_matrix = compute_diffusion_distance(laplacian, t)
        
        self.assertIsInstance(distance_matrix, np.ndarray)
        self.assertEqual(distance_matrix.shape, laplacian.shape)
        
        # Distance matrix should be symmetric
        self.assertTrue(np.allclose(distance_matrix, distance_matrix.T))
        
        # Diagonal elements should be zero
        self.assertTrue(np.allclose(np.diag(distance_matrix), 0.0))


if __name__ == '__main__':
    unittest.main()
