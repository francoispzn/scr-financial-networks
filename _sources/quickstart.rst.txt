Quick Start
===========

Launch the Dashboard
--------------------

.. code-block:: bash

   python dashboard/app.py
   # Open http://localhost:8050

The dashboard has 5 pages:

- **Network**: interactive interbank exposure graph with health indicators
- **Simulate**: run ABM stress scenarios in real-time
- **Spectral**: eigenvalue spectrum, diffusion distances, coarse-graining
- **Evolve**: train GNN from the UI with live progress
- **AI/Data**: LLM-powered risk analysis and data fetching

Train a GNN from the UI
------------------------

1. Navigate to **Evolve** → select **Real Market (yfinance)** → click **Load Data**
2. Adjust hyperparameters (hidden dim, GCN/LSTM layers, epochs, learning rate, dropout)
3. Click **Train GNN** — watch live progress with loss, RAM usage, and ETA
4. Review training loss curve, test scatter plot, and spectral evolution predictions

Programmatic Usage
------------------

.. code-block:: python

   from scr_financial.abm.simulation import BankingSystemSimulation
   from scr_financial.network.spectral import compute_laplacian, eigendecomposition
   from scr_financial.network.coarse_graining import SpectralCoarseGraining
   from scr_financial.ml.gnn_predictor import GNNPredictor

   # Run ABM simulation (inputs from the real data pipeline)
   from dashboard.data_loader import load_simulation_inputs
   bank_data, network_data, system_indicators = load_simulation_inputs()
   sim = BankingSystemSimulation(bank_data, network_data, system_indicators)
   sim.run_simulation(100)

   # Spectral analysis
   adj = sim.get_adjacency_matrix()
   L = compute_laplacian(adj, normalized=True)
   eigenvalues, eigenvectors = eigendecomposition(L)

   # Coarse-grain the network
   scg = SpectralCoarseGraining.from_adjacency(adj, list(sim.banks.keys()))
   scg.coarse_grain()
   accuracy = scg.compute_reconstruction_accuracy(time_steps=15)

   # Train GNN on market data
   from dashboard.data_api import build_daily_graph_snapshots
   snapshots = build_daily_graph_snapshots(lookback_years=3)
   predictor = GNNPredictor(seq_len=10, hidden_dim=64, num_gcn_layers=3)
   predictor.train(snapshots, epochs=200, lr=3e-3)
   predictions = predictor.predict(snapshots, steps=30)
