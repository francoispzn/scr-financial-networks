Black Week Simulation Example
============================

This example demonstrates how to use the SCR-Financial-Networks framework to simulate the "Black Week" of the 2008 financial crisis, focusing on interbank contagion dynamics.

Overview
--------

The 2008 financial crisis reached a critical point during the week of September 15-19, 2008, often referred to as "Black Week." During this period, Lehman Brothers filed for bankruptcy, AIG received a government bailout, and financial markets experienced extreme stress.

This example shows how to:

1. Set up the SCR-ABM framework with historical data
2. Define shock scenarios based on actual events
3. Run simulations with both full and coarse-grained networks
4. Analyze and visualize the results

Prerequisites
------------

Before running this example, ensure you have:

- Installed the SCR-Financial-Networks package
- Downloaded the required historical data (or use the provided sample data)
- Set up the conda environment as described in the installation guide

Code Example
-----------

.. code-block:: python

    import scr_financial as scrf
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from datetime import datetime, timedelta

    # Initialize data preprocessor
    preprocessor = scrf.data.DataPreprocessor(
        start_date='2008-01-01',
        end_date='2008-12-31'
    )

    # Load bank data for selected European banks
    preprocessor.load_bank_node_data({
        'solvency': 'EBA_transparency',
        'liquidity': 'EBA_aggregated',
        'market_risk': 'NYU_VLAB'
    })

    # Load network data
    preprocessor.load_interbank_exposures('ECB_TARGET2')

    # Initialize framework
    framework = scrf.SCG_ABM_Framework(preprocessor)

    # Set up for Black Week 2008 analysis
    black_week_start = '2008-09-15'  # Lehman Brothers bankruptcy
    framework.initialize_for_timepoint(black_week_start)

    # Define shocks based on historical events
    shocks = {
        1: {'Lehman_Brothers': {'CET1_ratio': -2.0, 'LCR': -30}},  # Day 1: Lehman shock
        3: {'AIG': {'CET1_ratio': -1.5, 'LCR': -25}},              # Day 3: AIG bailout
        5: {'system': {'funding_stress': 0.2}}                      # Day 5: General funding stress
    }

    # Run simulation with full network
    print("Running simulation with full network...")
    full_results = framework.run_simulation(10, shocks)

    # Run simulation with coarse-grained network
    print("Running simulation with coarse-grained network...")
    cg_results, clusters = framework.run_coarse_grained_simulation(10, shocks)

    # Compare results
    comparison = framework.compare_full_vs_coarse_grained(10, shocks)

    # Visualize results
    scrf.utils.visualize_results(full_results, cg_results,
