"""
Spectral coarse-graining for financial networks.

This module implements spectral coarse-graining techniques for simplifying
financial networks while preserving their essential structural and dynamic properties.
"""

import numpy as np
import networkx as nx
import scipy.sparse as sp
import scipy.linalg as la
from typing import Dict, List, Optional, Tuple, Union
from sklearn.cluster import SpectralClustering

from .builder import FinancialNetworkBuilder


class SpectralCoarseGraining:
    """
    Implements spectral coarse-graining for financial networks.
    
    Parameters
    ----------
    network_builder : FinancialNetworkBuilder
        Network builder containing the network to coarse-grain
    
    Attributes
    ----------
    coarse_grained_laplacian : numpy.ndarray
        Coarse-grained Laplacian matrix
    coarse_grained_adjacency : numpy.ndarray
        Coarse-grained adjacency matrix
    rescaled_adjacency : numpy.ndarray
        Rescaled coarse-grained adjacency matrix
    clusters : numpy.ndarray
        Cluster assignments for nodes
    """
    
    def __init__(self, network_builder: FinancialNetworkBuilder):
        """Initialize the coarse-graining with a network builder."""
        self.network_builder = network_builder
        self.coarse_grained_laplacian = None
        self.coarse_grained_adjacency = None
        self.rescaled_adjacency = None
        self.clusters = None
        self.cluster_mapping = None
        self.coarse_grained_graph = None
    
    def coarse_grain(self, k: Optional[int] = None) -> np.ndarray:
        """
        Perform spectral coarse-graining using k eigenmodes.
        
        Parameters
        ----------
        k : int, optional
            Number of eigenmodes to use, by default None
            If None, determined automatically using spectral gap
            
        Returns
        -------
        numpy.ndarray
            Coarse-grained Laplacian matrix
        """
        if self.network_builder.eigenvalues is None or self.network_builder.eigenvectors is None:
            self.network_builder.spectral_analysis()
        
        eigenvalues = self.network_builder.eigenvalues
        eigenvectors = self.network_builder.eigenvectors
        
        if k is None:
            # Use spectral gap to determine k
            k, _ = self.network_builder.find_spectral_gap()
        
        # Construct coarse-grained Laplacian using first k eigenmodes
        L_cg = np.zeros_like(self.network_builder.laplacian.todense())
        for alpha in range(k+1):  # Include eigenmode 0 to k
            lambda_alpha = eigenvalues[alpha]
            u_alpha = eigenvectors[:, alpha].reshape(-1, 1)
            L_cg += lambda_alpha * (u_alpha @ u_alpha.T)
            
        self.coarse_grained_laplacian = L_cg
        
        # Extract adjacency matrix from Laplacian
        # For a normalized Laplacian L = I - D^(-1/2)AD^(-1/2)
        # We need to reverse this to get A
        D = np.diag(np.diag(L_cg))
        self.coarse_grained_adjacency = D - L_cg
        
        return self.coarse_grained_laplacian
    
    def rescale(self, lambda_k: Optional[float] = None) -> np.ndarray:
        """
        Rescale the coarse-grained adjacency matrix.
        
        Parameters
        ----------
        lambda_k : float, optional
            Eigenvalue to use for rescaling, by default None
            If None, uses the k-th eigenvalue where k is from find_spectral_gap
            
        Returns
        -------
        numpy.ndarray
            Rescaled coarse-grained adjacency matrix
        """
        if self.coarse_grained_adjacency is None:
            raise ValueError("Coarse-grained adjacency not computed. Call coarse_grain first.")
        
        if lambda_k is None:
            k, _ = self.network_builder.find_spectral_gap()
            lambda_k = self.network_builder.eigenvalues[k]
        
        # Rescale by 1/lambda_k to preserve diffusion dynamics
        self.rescaled_adjacency = (1/lambda_k) * self.coarse_grained_adjacency
        
        return self.rescaled_adjacency
    
    def identify_clusters(self, n_clusters: Optional[int] = None) -> np.ndarray:
        """
        Identify clusters based on eigenvectors.
        
        Parameters
        ----------
        n_clusters : int, optional
            Number of clusters, by default None
            If None, determined from spectral gap
            
        Returns
        -------
        numpy.ndarray
            Cluster assignments for nodes
        """
        if self.network_builder.eigenvectors is None:
            self.network_builder.spectral_analysis()
        
        if n_clusters is None:
            k, _ = self.network_builder.find_spectral_gap()
            n_clusters = k + 1  # Use k+1 clusters
        
        # Extract relevant eigenvectors
        eigenvectors = self.network_builder.eigenvectors[:, 1:n_clusters+1]  # Skip first eigenvector
        
        # Apply k-means clustering
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.clusters = kmeans.fit_predict(eigenvectors)
        
        # Create mapping from nodes to clusters
        nodes = list(self.network_builder.G.nodes())
        self.cluster_mapping = {nodes[i]: self.clusters[i] for i in range(len(nodes))}
        
        return self.clusters
    
    def create_coarse_grained_graph(self) -> nx.Graph:
        """
        Create a coarse-grained graph based on identified clusters.
        
        Returns
        -------
        networkx.Graph
            Coarse-grained graph
        """
        if self.clusters is None:
            self.identify_clusters()
        
        # Create a new graph with clusters as nodes
        G_cg = nx.DiGraph()
        
        # Add cluster nodes
        n_clusters = len(set(self.clusters))
        for i in range(n_clusters):
            G_cg.add_node(f"cluster_{i}")
        
        # Compute inter-cluster weights
        original_graph = self.network_builder.G
        nodes = list(original_graph.nodes())
        
        for i in range(n_clusters):
            for j in range(n_clusters):
                if i != j:
                    # Find nodes in each cluster
                    nodes_i = [nodes[k] for k in range(len(nodes)) if self.clusters[k] == i]
                    nodes_j = [nodes[k] for k in range(len(nodes)) if self.clusters[k] == j]
                    
                    # Compute total weight between clusters
                    total_weight = 0
                    for source in nodes_i:
                        for target in nodes_j:
                            if original_graph.has_edge(source, target):
                                total_weight += original_graph[source][target]['weight']
                    
                    # Add edge if there's a connection
                    if total_weight > 0:
                        G_cg.add_edge(f"cluster_{i}", f"cluster_{j}", weight=total_weight)
        
        # Compute aggregate node attributes
        for i in range(n_clusters):
            nodes_i = [nodes[k] for k in range(len(nodes)) if self.clusters[k] == i]
            
            # Aggregate node attributes
            for attr in original_graph.nodes[nodes[0]]:
                # Skip non-numeric attributes
                if not isinstance(original_graph.nodes[nodes[0]][attr], (int, float)):
                    continue
                
                # Compute average
                avg_value = np.mean([original_graph.nodes[node][attr] for node in nodes_i 
                                    if attr in original_graph.nodes[node]])
                
                G_cg.nodes[f"cluster_{i}"][attr] = avg_value
            
            # Store original nodes in this cluster
            G_cg.nodes[f"cluster_{i}"]["original_nodes"] = nodes_i
            G_cg.nodes[f"cluster_{i}"]["size"] = len(nodes_i)
        
        self.coarse_grained_graph = G_cg
        
        return G_cg
    
    def map_node_to_cluster(self, node: str) -> str:
        """
        Map an original node to its cluster.
        
        Parameters
        ----------
        node : str
            Original node identifier
            
        Returns
        -------
        str
            Cluster identifier
        """
        if self.cluster_mapping is None:
            raise ValueError("Clusters not identified. Call identify_clusters first.")
        
        if node not in self.cluster_mapping:
            raise ValueError(f"Node {node} not found in the network")
        
        return f"cluster_{self.cluster_mapping[node]}"
    
    def compute_coarse_graining_error(self) -> float:
        """
        Compute the error introduced by coarse-graining.
        
        Returns
        -------
        float
            Relative error in eigenvalues
        """
        if self.coarse_grained_laplacian is None:
            raise ValueError("Coarse-grained Laplacian not computed. Call coarse_grain first.")
        
        # Get original eigenvalues
        orig_eigenvalues = self.network_builder.eigenvalues
        
        # Compute eigenvalues of coarse-grained Laplacian
        cg_eigenvalues, _ = la.eigh(self.coarse_grained_laplacian)
        
        # Determine k
        k, _ = self.network_builder.find_spectral_gap()
        
        # Compare first k+1 eigenvalues
        error = np.linalg.norm(orig_eigenvalues[:k+1] - cg_eigenvalues[:k+1]) / np.linalg.norm(orig_eigenvalues[:k+1])
        
        return error
    
    def compare_diffusion_dynamics(self, time_steps: int = 10) -> List[float]:
        """
        Compare diffusion dynamics between original and coarse-grained networks.
        
        Parameters
        ----------
        time_steps : int, optional
            Number of time steps to simulate, by default 10
            
        Returns
        -------
        list
            List of errors at each time step
        """
        if self.coarse_grained_laplacian is None:
            raise ValueError("Coarse-grained Laplacian not computed. Call coarse_grain first.")
        
        # Original Laplacian
        L_orig = self.network_builder.laplacian.todense()
        
        # Coarse-grained Laplacian
        L_cg = self.coarse_grained_laplacian
        
        # Initial state (uniform distribution)
        p0 = np.ones(L_orig.shape[0]) / L_orig.shape[0]
        
        # Simulate diffusion on both networks
        errors = []
        for t in range(1, time_steps + 1):
            # Original diffusion
            pt_orig = np.exp(-t * L_orig) @ p0
            
            # Coarse-grained diffusion
            pt_cg = np.exp(-t * L_cg) @ p0
            
            # Compute error
            error = np.linalg.norm(pt_orig - pt_cg) / np.linalg.norm(pt_orig)
            errors.append(error)
        
        return errors
