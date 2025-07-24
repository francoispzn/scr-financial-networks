Installation
============

Requirements
------------

- Python 3.9+
- PyTorch 2.0+
- PyTorch Geometric 2.6+

Using Conda (recommended)
-------------------------

.. code-block:: bash

   git clone https://github.com/francoispzn/scr-financial-networks.git
   cd scr-financial-networks
   conda env create -f environment.yml
   conda activate systemic_risk
   pip install -e .

Using pip
---------

.. code-block:: bash

   git clone https://github.com/francoispzn/scr-financial-networks.git
   cd scr-financial-networks
   pip install -e ".[dev]"

Core Dependencies
-----------------

- ``numpy``, ``scipy``, ``pandas`` — numerical computing
- ``networkx`` — graph construction and analysis
- ``torch``, ``torch-geometric`` — GNN training
- ``dash``, ``plotly`` — interactive dashboard
- ``yfinance``, ``requests`` — market data fetching
