SCR-Financial-Networks Documentation
===================================

A Python framework for analyzing financial networks using Spectral Coarse-Graining (SCG) and Agent-Based Modeling (ABM).

Overview
--------

This project implements a hybrid approach combining Spectral Coarse-Graining (SCG) with Agent-Based Modeling (ABM) to analyze interbank contagion dynamics among European banks. The framework enables the study of systemic risk, contagion pathways, and financial stability by preserving both macro-level network effects and micro-level bank behaviors.

The methodology is based on the theoretical framework described in "Spectral Coarse-Graining to Financial Networks" and addresses the research question posed in the "Contingency BW 9-08 x10-15" preliminary report.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   examples/index
   api/index
   theory/index
   contributing
   changelog

Installation
-----------

Using Conda:

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/username/scr-financial-networks.git
   cd scr-financial-networks

   # Create and activate the conda environment
   conda env create -f environment.yml
   conda activate scr-financial

   # Install the package in development mode
   pip install -e .

Using pip:

.. code-block:: bash

   pip install scr-financial-networks

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
