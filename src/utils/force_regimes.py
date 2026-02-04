# src/utils/force_regimes.py
import pandas as pd
import numpy as np
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import config

def force_3_regimes():
    print("☢️ FORCING 3 REGIMES (Predictive Shift Mode)...")
    
    # 1. Load Data
    data_path = os.path.join(config.DATA_PROCESSED, 'features.csv')
    if not os.path.exists(data_path):
        print("❌ Error: features.csv not found. Run engineer.py first.")
        return

    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    
    # 2. Calculate Thresholds based on NEXT DAY'S Return
    # We want to label Today based on what Tomorrow does.
    # Create a temporary column for Next Day Return
    df['Next_Day_Ret'] = df['SPY_Log_Ret'].shift(-1)
    
    # Drop the last row since it has no 'Tomorrow'
    df.dropna(subset=['Next_Day_Ret'], inplace=True)
    
    ret_col = 'Next_Day_Ret'
    
    low_cutoff = df[ret_col].quantile(0.33)
    high_cutoff = df[ret_col].quantile(0.66)
    
    print(f"   Bear/Sideways Cutoff (Next Day): {low_cutoff:.5f}")
    print(f"   Sideways/Bull Cutoff (Next Day): {high_cutoff:.5f}")
    
    # 3. Assign Labels based on TOMORROW'S Return
    # 0 = Bear (Tomorrow is Crash)
    # 1 = Sideways (Tomorrow is Flat)
    # 2 = Bull (Tomorrow is Pump)
    
    conditions = [
        (df[ret_col] <= low_cutoff),
        (df[ret_col] > low_cutoff) & (df[ret_col] <= high_cutoff),
        (df[ret_col] > high_cutoff)
    ]
    choices = [0, 1, 2]
    
    df['Regime'] = np.select(conditions, choices, default=1)
    
    # High Confidence for training
    df['Regime_Confidence'] = 0.99
    
    # 4. Cleanup
    # We don't need the temporary column anymore
    df.drop(columns=['Next_Day_Ret'], inplace=True)

    # 5. Save
    save_path = os.path.join(config.DATA_PROCESSED, 'market_data_with_regimes.csv')
    df.to_csv(save_path)
    
    # Verify
    counts = df['Regime'].value_counts().sort_index()
    print("\n✅ New Predictive Regime Distribution:")
    print(counts)
    print(f"💾 Saved to {save_path}")

if __name__ == "__main__":
    force_3_regimes()