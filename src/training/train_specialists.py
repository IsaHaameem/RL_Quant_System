# src/training/train_specialists.py
import pandas as pd
import numpy as np
import os
import sys
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import config
from src.training.trading_env import RegimeAwareTradingEnv

def train_specialists():
    print("🚀 Starting Mixture-of-Experts Training (Robust Mode)...")
    
    # 1. Load Data
    data_path = os.path.join(config.DATA_PROCESSED, 'market_data_with_regimes.csv')
    df = pd.read_csv(data_path)
    
    specialists_dir = os.path.join(config.MODELS_DIR, 'specialists')
    os.makedirs(specialists_dir, exist_ok=True)
    
    # 2. Train One Agent Per Regime
    for regime_id in range(config.N_REGIMES):
        print(f"\n👨‍🏫 Training Specialist for Regime {regime_id}...")
        
        # Filter Data for this regime
        regime_df = df[df['Regime'] == regime_id].reset_index(drop=True)
        data_len = len(regime_df)
        
        print(f"📊 Original Dataset size: {data_len} bars")
        
        # --- FIX: SYNTHETIC DATA FALLBACK ---
        # If the HMM failed to find a regime (e.g., Bull is empty), we create it manually
        # based on Returns. This guarantees the agent exists.
        if data_len < 100:
            print(f"⚠️ Regime {regime_id} is empty/sparse. Creating SYNTHETIC training data...")
            
            if regime_id == 2: # Bull/Trend (Highest ID)
                # Fallback: Train on the top 33% highest return days
                print("   -> Selecting Top 33% Best Market Days for Bull Training")
                regime_df = df.sort_values(by='SPY_Log_Ret', ascending=False).head(len(df) // 3).reset_index(drop=True)
                
            elif regime_id == 0: # Bear/Crash (Lowest ID)
                # Fallback: Train on the bottom 33% worst return days
                print("   -> Selecting Bottom 33% Worst Market Days for Bear Training")
                regime_df = df.sort_values(by='SPY_Log_Ret', ascending=True).head(len(df) // 3).reset_index(drop=True)
            
            else: # Sideways
                # Fallback: Random sample
                regime_df = df.sample(n=1000).reset_index(drop=True)
                
            print(f"   🔧 Synthetic Dataset size: {len(regime_df)} bars")

        # --- OVERSAMPLING ---
        # Even if not empty, if it's small (<500), duplicate it so the AI learns
        if len(regime_df) < 500:
            print(f"   🔧 Oversampling small dataset (x5)...")
            regime_df = pd.concat([regime_df] * 5, ignore_index=True)
            
        # Setup Environment
        train_env = DummyVecEnv([lambda: RegimeAwareTradingEnv(regime_df)])
        
        # Hyperparameters
        entropy = config.ENTROPY_BETA
        # Force Bear agent (0) and Bull agent (2) to be more aggressive
        if regime_id == 0 or regime_id == 2: 
            entropy = 0.05 
        
        model = PPO(
            "MlpPolicy", 
            train_env, 
            verbose=0,
            learning_rate=config.LEARNING_RATE,
            gamma=config.GAMMA,
            ent_coef=entropy, 
            device="cuda" # Fallback to cpu automatically
        )
        
        # Train
        model.learn(total_timesteps=50000) 
        
        save_path = os.path.join(specialists_dir, f'agent_regime_{regime_id}.zip')
        model.save(save_path)
        print(f"✅ Saved Specialist {regime_id} to {save_path}")

if __name__ == "__main__":
    train_specialists()