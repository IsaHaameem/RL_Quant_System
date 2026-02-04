# src/utils/explain.py
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import config

def plot_regimes():
    print("💡 Generating Explainability Charts...")
    
    # Load Data
    data_path = os.path.join(config.DATA_PROCESSED, 'market_data_with_regimes.csv')
    df = pd.read_csv(data_path)
    
    # We only want the Test Period (last 20% of data)
    split_index = int(len(df) * 0.8)
    test_df = df.iloc[split_index:].reset_index(drop=True)
    
    # Plot Price colored by Regime
    plt.figure(figsize=(15, 7))
    
    # Extract Price and Regime
    price = test_df['SPY_Close']
    regime = test_df['Regime']
    
    # Plot the base price line (faint)
    plt.plot(price, color='black', alpha=0.2, label='Market Price')
    
    # Scatter plot for Regimes
    # Regime 0 (Bear/Crash) -> Red
    # Regime 1 (Sideways) -> Yellow/Orange
    # Regime 2 (Bull) -> Green
    
    # Note: Adjust these colors based on your specific HMM output from Step 6
    # You might need to swap colors if Regime 0 was actually Bull in your run.
    colors = ['red', 'orange', 'green'] 
    labels = ['Regime 0', 'Regime 1', 'Regime 2']
    
    for r in range(3):
        idx = regime == r
        if idx.any():
            plt.scatter(test_df.index[idx], price[idx], 
                        color=colors[r], s=10, label=f'Regime {r}', alpha=0.6)
            
    plt.title('Explainability: Market Regimes Detected by AI')
    plt.xlabel('Trading Days')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path = os.path.join(config.PROJECT_ROOT, 'explainability_regimes.png')
    plt.savefig(save_path)
    print(f"📊 Explainability Chart saved to: {save_path}")
    plt.show()

if __name__ == "__main__":
    plot_regimes()