# src/features/engineer.py
import pandas as pd
import numpy as np
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import config

class FeatureEngineer:
    def __init__(self, input_path=None):
        if input_path is None:
            self.input_path = os.path.join(config.DATA_RAW, 'market_data.csv')
        else:
            self.input_path = input_path

    def compute_technical_indicators(self, df, ticker):
        """
        Computes Log Returns, Rolling Volatility, and RSI for a specific ticker.
        """
        price_col = f"{ticker}_Close"
        
        # 1. Log Returns (Stationary) - The primary input for RL
        # ln(P_t / P_{t-1})
        df[f'{ticker}_Log_Ret'] = np.log(df[price_col] / df[price_col].shift(1))
        
        # 2. Rolling Volatility (for Regime Detection)
        # Standard deviation of returns over last 20 periods
        df[f'{ticker}_Vol_20'] = df[f'{ticker}_Log_Ret'].rolling(window=20).std()
        
        # 3. Simple Moving Average Momentum
        # Price / SMA_50 - 1 (Normalized distance from mean)
        sma_50 = df[price_col].rolling(window=50).mean()
        df[f'{ticker}_Mom_50'] = (df[price_col] / sma_50) - 1
        
        return df

    def process_data(self):
        print("⚙️  Starting Feature Engineering...")
        if not os.path.exists(self.input_path):
            raise FileNotFoundError(f"❌ Raw data not found at {self.input_path}. Run downloader.py first.")

        df = pd.read_csv(self.input_path, index_col=0, parse_dates=True)
        
        # Apply engineering to all tickers
        # NOTE: We treat SPY as the 'Market' for general regime detection
        for ticker in config.TICKERS:
            df = self.compute_technical_indicators(df, ticker)
            
        # Clean up NaNs created by rolling windows
        df.dropna(inplace=True)
        
        # Save processed data
        save_path = os.path.join(config.DATA_PROCESSED, 'features.csv')
        df.to_csv(save_path)
        print(f"✅ Features engineered and saved to: {save_path}")
        print(f"   New Columns: {[c for c in df.columns if 'Log_Ret' in c]}")
        return df

if __name__ == "__main__":
    engineer = FeatureEngineer()
    engineer.process_data()