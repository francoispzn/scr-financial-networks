"""
VAE Financial Features Example

This script demonstrates how to use the Variational Autoencoder (VAE) module
for dimensionality reduction and feature extraction from financial data.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import seaborn as sns

# Add parent directory to path to import scr_financial
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import scr_financial as scrf
from scr_financial.vae.model import FinancialVAE
from scr_financial.vae.training import train_vae, evaluate_vae
from scr_financial.data.preprocessor import DataPreprocessor
from scr_financial.data.collectors.market_collector import MarketDataCollector


def main():
    """Run the VAE financial features example."""
    print("Starting VAE Financial Features Example")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Define European banks to include in the analysis
    bank_list = [
        "DE_DBK",  # Deutsche Bank
        "FR_BNP",  # BNP Paribas
        "ES_SAN",  # Santander
        "IT_UCG",  # UniCredit
        "NL_ING",  # ING
        "SE_NDA",  # Nordea
        "CH_UBS",  # UBS
        "UK_BARC", # Barclays
        "UK_HSBC", # HSBC
        "FR_ACA"   # Credit Agricole
    ]
    
    # Initialize data preprocessor
    print("Initializing data preprocessor...")
    preprocessor = DataPreprocessor(
        start_date='2015-01-01',
        end_date='2020-12-31',
        bank_list=bank_list
    )
    
    # Load financial data
    print("Loading financial data...")
    preprocessor.load_bank_node_data({
        'solvency': 'EBA_transparency',
        'liquidity': 'EBA_aggregated',
        'market_risk': 'NYU_VLAB'
    })
    
    # Initialize market data collector for additional features
    market_collector = MarketDataCollector()
    
    # Collect CDS data
    print("Collecting CDS data...")
    cds_data = market_collector.collect_cds_data(
        preprocessor.start_date,
        preprocessor.end_date,
        bank_list
    )
    
    # Collect equity data
    print("Collecting equity data...")
    equity_data = market_collector.collect_equity_data(
        preprocessor.start_date,
        preprocessor.end_date,
        bank_list
    )
    
    # Create a combined dataset with all features
    print("Creating combined dataset...")
    
    # Get unique dates from CDS data
    dates = sorted(cds_data['date'].unique())
    
    # Create empty DataFrame to store combined data
    combined_data = []
    
    for date in dates:
        # Get CDS data for this date
        date_cds = cds_data[cds_data['date'] == date]
        
        # Get equity data for this date
        date_equity = equity_data[equity_data['date'] == date]
        
        # Get bank data for this date
        date_bank_data = preprocessor.get_data_for_timepoint(date)
        
        for bank_id in bank_list:
            # Initialize row with date and bank_id
            row = {'date': date, 'bank_id': bank_id}
            
            # Add CDS data
            bank_cds = date_cds[date_cds['bank_id'] == bank_id]
            if not bank_cds.empty:
                row['CDS_5yr'] = bank_cds['CDS_5yr'].values[0]
            
            # Add equity data
            bank_equity = date_equity[date_equity['bank_id'] == bank_id]
            if not bank_equity.empty:
                row['price'] = bank_equity['price'].values[0]
                row['volume'] = bank_equity['volume'].values[0]
            
            # Add solvency data
            if 'solvency' in date_bank_data['node_data']:
                solvency_data = date_bank_data['node_data']['solvency']
                if isinstance(solvency_data, pd.DataFrame):
                    bank_solvency = solvency_data[solvency_data['bank_id'] == bank_id]
                    if not bank_solvency.empty:
                        for col in bank_solvency.columns:
                            if col not in ['date', 'bank_id']:
                                row[col] = bank_solvency[col].values[0]
            
            # Add liquidity data
            if 'liquidity' in date_bank_data['node_data']:
                liquidity_data = date_bank_data['node_data']['liquidity']
                if isinstance(liquidity_data, pd.DataFrame):
                    bank_liquidity = liquidity_data[liquidity_data['bank_id'] == bank_id]
                    if not bank_liquidity.empty:
                        for col in bank_liquidity.columns:
                            if col not in ['date', 'bank_id']:
                                row[col] = bank_liquidity[col].values[0]
            
            # Add market risk data
            if 'market_risk' in date_bank_data['node_data']:
                market_risk_data = date_bank_data['node_data']['market_risk']
                if isinstance(market_risk_data, pd.DataFrame):
                    bank_market_risk = market_risk_data[market_risk_data['bank_id'] == bank_id]
                    if not bank_market_risk.empty:
                        for col in bank_market_risk.columns:
                            if col not in ['date', 'bank_id']:
                                row[col] = bank_market_risk[col].values[0]
            
            # Add system indicators
            for indicator, value in date_bank_data['system_data'].items():
                row[f'system_{indicator}'] = value
            
            # Add row to combined data
            combined_data.append(row)
    
    # Convert to DataFrame
    combined_df = pd.DataFrame(combined_data)
    
    # Fill missing values
    combined_df = combined_df.fillna(method='ffill').fillna(method='bfill').fillna(0)
    
    print(f"Combined dataset shape: {combined_df.shape}")
    print(f"Features: {combined_df.columns.tolist()}")
    
    # Select features for VAE
    feature_columns = [
        'CDS_5yr', 'price', 'volume', 'CET1_ratio', 'Tier1_leverage_ratio',
        'LCR', 'NSFR', 'SRISK', 'total_assets', 'risk_weighted_assets',
        'system_CISS', 'system_funding_stress', 'system_credit_to_GDP_gap'
    ]
    
    # Filter columns that exist in the DataFrame
    feature_columns = [col for col in feature_columns if col in combined_df.columns]
    
    # Prepare data for VAE
    print("Preparing data for VAE...")
    
    # Extract features
    features = combined_df[feature_columns].values
    
    # Scale features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    # Split data into train and test sets
    train_size = int(0.8 * len(scaled_features))
    train_data = scaled_features[:train_size]
    test_data = scaled_features[train_size:]
    
    # Convert to torch tensors
    train_tensor = torch.FloatTensor(train_data)
    test_tensor = torch.FloatTensor(test_data)
    
    # Define VAE model
    input_dim = train_data.shape[1]
    hidden_dims = [64, 32, 16]
    latent_dim = 2  # Using 2D latent space for easy visualization
    
    print(f"Creating VAE model with input dimension {input_dim} and latent dimension {latent_dim}")
    model = FinancialVAE(input_dim, hidden_dims, latent_dim)
    
    # Train the model
    print("Training VAE model...")
    history = train_vae(
        model=model,
        data=train_tensor,
        batch_size=64,
        epochs=100,
        learning_rate=1e-3,
        device=device,
        verbose=True
    )
    
    # Create output directory if it doesn't exist
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['total_loss'], label='Total Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['reconstruction_loss'], label='Reconstruction Loss')
    plt.plot(history['kld_loss'], label='KLD Loss')
    plt.title('Loss Components')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "vae_training_history.png"), dpi=300, bbox_inches='tight')
    
    # Evaluate the model
    print("Evaluating VAE model...")
    eval_results = evaluate_vae(
        model=model,
        data=test_tensor,
        batch_size=64,
        device=device
    )
    
    print(f"Test Loss: {eval_results['loss']:.4f}")
    print(f"Test Reconstruction Loss: {eval_results['reconstruction_loss']:.4f}")
    print(f"Test KLD Loss: {eval_results['kld_loss']:.4f}")
    
    # Extract latent representations for all data
    print("Extracting latent representations...")
    model.eval()
    with torch.no_grad():
        all_tensor = torch.FloatTensor(scaled_features).to(device)
        latent_representations = model.encode(all_tensor).cpu().numpy()
    
    # Add latent representations to DataFrame
    combined_df['latent_1'] = latent_representations[:, 0]
    combined_df['latent_2'] = latent_representations[:, 1]
    
    # Visualize latent space
    print("Visualizing latent space...")
    plt.figure(figsize=(12, 10))
    
    # Color points by bank
    unique_banks = combined_df['bank_id'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_banks)))
    bank_to_color = {bank: color for bank, color in zip(unique_banks, colors)}
    
    # Plot each bank with a different color
    for bank_id in unique_banks:
        bank_data = combined_df[combined_df['bank_id'] == bank_id]
        plt.scatter(
            bank_data['latent_1'],
            bank_data['latent_2'],
            c=[bank_to_color[bank_id]],
            label=bank_id,
            alpha=0.7,
            s=50
        )
    
    plt.title('Latent Space Representation of Bank Features')
    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "latent_space_by_bank.png"), dpi=300, bbox_inches='tight')
    
    # Visualize latent space by time
    plt.figure(figsize=(12, 10))
    
    # Convert date to ordinal for coloring
    combined_df['date_ordinal'] = pd.to_datetime(combined_df['date']).apply(lambda x: x.toordinal())
    min_date = combined_df['date_ordinal'].min()
    max_date = combined_df['date_ordinal'].max()
    
    # Create a scatter plot with points colored by date
    scatter = plt.scatter(
        combined_df['latent_1'],
        combined_df['latent_2'],
        c=combined_df['date_ordinal'],
        cmap='viridis',
        alpha=0.7,
        s=50
    )
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Date')
    
    # Set colorbar ticks to actual dates
    date_ticks = np.linspace(min_date, max_date, 5)
    cbar.set_ticks(date_ticks)
    cbar.set_ticklabels([pd.Timestamp.fromordinal(int(tick)).strftime('%Y-%m-%d') for tick in date_ticks])
    
    plt.title('Latent Space Representation by Time')
    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "latent_space_by_time.png"), dpi=300, bbox_inches='tight')
    
    # Visualize latent space by CET1 ratio
    if 'CET1_ratio' in combined_df.columns:
        plt.figure(figsize=(12, 10))
        
        # Create a scatter plot with points colored by CET1 ratio
        scatter = plt.scatter(
            combined_df['latent_1'],
            combined_df['latent_2'],
            c=combined_df['CET1_ratio'],
            cmap='RdYlGn',
            alpha=0.7,
            s=50
        )
        
        # Add colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label('CET1 Ratio (%)')
        
        plt.title('Latent Space Representation by CET1 Ratio')
        plt.xlabel('Latent Dimension 1')
        plt.ylabel('Latent Dimension 2')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, "latent_space_by_cet1.png"), dpi=300, bbox_inches='tight')
    
    # Visualize latent space by CDS spread
    if 'CDS_5yr' in combined_df.columns:
        plt.figure(figsize=(12, 10))
        
        # Create a scatter plot with points colored by CDS spread
        scatter = plt.scatter(
            combined_df['latent_1'],
            combined_df['latent_2'],
            c=combined_df['CDS_5yr'],
            cmap='YlOrRd',
            alpha=0.7,
            s=50
        )
        
        # Add colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label('5-Year CDS Spread (bps)')
        
        plt.title('Latent Space Representation by CDS Spread')
        plt.xlabel('Latent Dimension 1')
        plt.ylabel('Latent Dimension 2')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, "latent_space_by_cds.png"), dpi=300, bbox_inches='tight')
    
    # Analyze feature correlations with latent dimensions
    print("Analyzing feature correlations with latent dimensions...")
    
    # Calculate correlations
    correlations = combined_df[feature_columns + ['latent_1', 'latent_2']].corr()
    
    # Extract correlations with latent dimensions
    latent_correlations = correlations[['latent_1', 'latent_2']].iloc[:-2]
    
    # Plot correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        latent_correlations,
        annot=True,
        cmap='coolwarm',
        vmin=-1,
        vmax=1,
        center=0,
        fmt='.2f'
    )
    plt.title('Feature Correlations with Latent Dimensions')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "latent_correlations.png"), dpi=300, bbox_inches='tight')
    
    # Generate new samples
    print("Generating new samples...")
    num_samples = 10
    generated_samples = model.sample(num_samples, device=device).cpu().numpy()
    
    # Inverse transform to original scale
    generated_samples = scaler.inverse_transform(generated_samples)
    
    # Create DataFrame with generated samples
    generated_df = pd.DataFrame(generated_samples, columns=feature_columns)
    
    # Save generated samples to CSV
    generated_df.to_csv(os.path.join(output_dir, "generated_samples.csv"), index=False)
    
    print("Generated sample features:")
    print(generated_df.head())
    
    # Save model
    print("Saving model...")
    torch.save(model.state_dict(), os.path.join(output_dir, "financial_vae_model.pth"))
    
    # Save latent representations
    latent_df = combined_df[['date', 'bank_id', 'latent_1', 'latent_2']].copy()
    latent_df.to_csv(os.path.join(output_dir, "latent_representations.csv"), index=False)
    
    print("\nVAE analysis complete. Results saved to the 'output' directory.")


if __name__ == "__main__":
    main()
