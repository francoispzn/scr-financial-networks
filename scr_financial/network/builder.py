"""
Network builder for financial network analysis.

This module provides the FinancialNetworkBuilder class for constructing
financial networks from preprocessed data.
"""

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.linalg as la
from typing import Dict, List, Optional, Tuple, Union

from ..data.preprocessor import DataPreprocessor


class FinancialNetworkBuilder:
    """
    Constructs financial networks from preprocessed data.
    
    Parameters
    ----------
    preprocessor : DataPreprocessor
        Preprocessor containing the data to build the network from
    
    Attributes
    ----------
    G : networkx.Graph
        NetworkX graph representing the financial network
    adjacency_matrix : scipy.sparse.csr_matrix
        Sparse adjacency matrix of the network
    laplacian : scipy.sparse.csr_matrix
        Graph Laplacian matrix
    eigenvalues : numpy.ndarray
        Eigenvalues of the Laplacian
    eigenvectors : numpy.ndarray
        Eigenvectors of the Laplacian
    """
    
    def __init__(self, preprocessor: DataPreprocessor):
        """Initialize the network builder with preprocessed data."""
        self.preprocessor = preprocessor
        self.G = None
        self.adjacency_matrix = None
        self.laplacian = None
        self.eigenvalues = None
        self.eigenvectors = None
    
    def construct_network(self, time_point: Union[str, pd.Timestamp], edge_weight_type: str = 'interbank_exposures') -> nx.Graph:
        """
        Construct network for a specific time point.
        
        Parameters
        ----------
        time_point : str or pd.Timestamp
            Date for which to construct the network
        edge_weight_type : str, optional
            Type of edge weight to use, by default 'interbank_exposures'
            
        Returns
        -------
        networkx.Graph
            Constructed financial network
        """
        # Get data for the specified time point
        data = self.preprocessor.get_data_for_timepoint(time_point)
        
        # Extract edge data
        if edge_weight_type not in data['edge_data']:
            raise ValueError(f"Edge weight type '{edge_weight_type}' not found in data")
        
        edge_data = data['edge_data'][edge_weight_type]
        
        # Create empty graph
        G = nx.DiGraph()
        
        # Add edges
        for _, row in edge_data.iterrows():
            G.add_edge(
                row['source'],
                row['target'],
                weight=row['weight']
            )
        
        # Add node attributes
        for category, node_data in data['node_data'].items():
            if isinstance(node_data, pd.DataFrame):
                for _, row in node_data.iterrows():
                    bank_id = row['bank_id']
                    if bank_id in G.nodes():
                        for col in node_data.columns:
                            if col != 'bank_id' and col != 'date':
                                G.nodes[bank_id][col] = row[col]
            else:
                # If node_data is a Series (single time point)
                for bank_id, value in node_data.items():
                    if bank_id in G.nodes():
                        G.nodes[bank_id][category] = value
        
        # Store the graph
        self.G = G
        
        # Create adjacency matrix
        self.adjacency_matrix = nx.to_scipy_sparse_array(G, weight='weight')
        
        return G
    
    def compute_laplacian(self, normalized: bool = True) -> sp.csr_matrix:
        """
        Compute the graph Laplacian.
        
        Parameters
        ----------
        normalized : bool, optional
            Whether to compute the normalized Laplacian, by default True
            
        Returns
        -------
        scipy.sparse.csr_matrix
            Graph Laplacian matrix
        """
        if self.G is None:
            raise ValueError("Network not constructed. Call construct_network first.")
        
        if normalized:
            self.laplacian = nx.normalized_laplacian_matrix(self.G, weight='weight')
        else:
            self.laplacian = nx.laplacian_matrix(self.G, weight='weight')
        
        return self.laplacian
    
    def spectral_analysis(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform spectral analysis of the Laplacian.
        
        Returns
        -------
        tuple
            Tuple containing eigenvalues and eigenvectors
        """
        if self.laplacian is None:
            raise ValueError("Laplacian not computed. Call compute_laplacian first.")
        
        # For small networks, we can use dense eigendecomposition
        L_dense = self.laplacian.todense()
        self.eigenvalues, self.eigenvectors = la.eigh(L_dense)
        
        return self.eigenvalues, self.eigenvectors
    
    def find_spectral_gap(self) -> Tuple[int, float]:
        """
        Identify the spectral gap for coarse-graining.
        
        Returns
        -------
        tuple
            Tuple containing the index of the gap and the gap size
        """
        if self.eigenvalues is None:
            raise ValueError("Spectral analysis not performed. Call spectral_analysis first.")
        
        # Compute differences between consecutive eigenvalues
        gaps = np.diff(self.eigenvalues)
        
        # Find the largest gap after the first few eigenvalues
        # Skip the first eigenvalue (which is 0 for connected graphs)
        k = np.argmax(gaps[1:]) + 1
        
        return k, gaps[k]
    
    def get_node_attribute_matrix(self, attribute: str) -> np.ndarray:
        """
        Get matrix of node attributes.
        
        Parameters
        ----------
        attribute : str
            Node attribute to extract
            
        Returns
        -------
        numpy.ndarray
            Matrix of node attributes
        """
        if self.G is None:
            raise ValueError("Network not constructed. Call construct_network first.")
        
        # Extract attribute for all nodes
        values = []
        for node in self.G.nodes():
            if attribute in self.G.nodes[node]:
                values.append(self.G.nodes[node][attribute])
            else:
                values.append(0)
        
        return np.array(values)
    
    def get_edge_weight_matrix(self) -> np.ndarray:
        """
        Get matrix of edge weights.
        
        Returns
        -------
        numpy.ndarray
            Matrix of edge weights
        """
        if self.adjacency_matrix is None:
            raise ValueError("Network not constructed. Call construct_network first.")
        
        return self.adjacency_matrix.todense()
    
    def compute_centrality_measures(self) -> Dict[str, Dict[str, float]]:
        """
        Compute various centrality measures for the network.
        
        Returns
        -------
        dict
            Dictionary mapping centrality types to dictionaries of node centralities
        """
        if self.G is None:
            raise ValueError("Network not constructed. Call construct_network first.")
        
        centrality_measures = {}
        
        # Degree centrality
        centrality_measures['degree'] = nx.degree_centrality(self.G)
        
        # Eigenvector centrality
        try:
            centrality_measures['eigenvector'] = nx.eigenvector_centrality(self.G, weight='weight')
        except:
            # May not converge for some networks
            centrality_measures['eigenvector'] = {node: 0 for node in self.G.nodes()}
        
        # Betweenness centrality
        centrality_measures['betweenness'] = nx.betweenness_centrality(self.G, weight='weight')
        
        # Closeness centrality
        centrality_measures['closeness'] = nx.closeness_centrality(self.G, distance='weight')
        
        # PageRank
        centrality_measures['pagerank'] = nx.pagerank(self.G, weight='weight')
        
        return centrality_measures
    
    def compute_community_structure(self, method: str = 'louvain') -> Dict[str, int]:
        """
        Compute community structure of the network.
        
        Parameters
        ----------
        method : str, optional
            Community detection method, by default 'louvain'
            
        Returns
        -------
        dict
            Dictionary mapping nodes to community IDs
        """
        if self.G is None:
            raise ValueError("Network not constructed. Call construct_network first.")
        
        if method == 'louvain':
            try:
                import community as community_louvain
                return community_louvain.best_partition(self.G.to_undirected(), weight='weight')
            except ImportError:
                print("python-louvain package not installed. Using spectral clustering instead.")
                method = 'spectral'
        
        if method == 'spectral':
            if self.eigenvalues is None:
                self.spectral_analysis()
            
            # Use spectral clustering
            from sklearn.cluster import SpectralClustering
            
            # Determine number of clusters from spectral gap
            k, _ = self.find_spectral_gap()
            n_clusters = k + 1  # Use k+1 clusters
            
            # Create adjacency matrix
            A = nx.to_numpy_array(self.G, weight='weight')
            
            # Apply spectral clustering
            clustering = SpectralClustering(
                n_clusters=n_clusters,
                affinity='precomputed',
                assign_labels='kmeans'
            )
            
            # Create affinity matrix from adjacency
            clustering.fit(A)
            
            # Map nodes to communities
            nodes = list(self.G.nodes())
            communities = {nodes[i]: clustering.labels_[i] for i in range(len(nodes))}
            
            return communities
        
        raise ValueError(f"Unknown community detection method: {method}")
