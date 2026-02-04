# src/backtesting/backtest.py
import pandas as pd
import numpy as np
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import config
from src.training.trading_env import RegimeAwareTradingEnv
from src.models.meta_controller import MetaController

def run_backtest():
    print("🧪 Starting Mixture-of-Experts Backtest...")
    
    # 1. Load Data & Preserve Dates
    data_path = os.path.join(config.DATA_PROCESSED, 'market_data_with_regimes.csv')
    # Load with Date as index
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    
    split_index = int(len(df) * 0.8)
    
    # CRITICAL FIX: reset_index() without drop=True keeps the 'Date' index as a column named 'Date' (or 'index')
    test_df = df.iloc[split_index:].reset_index()
    
    # Rename 'index' to 'Date' if necessary
    if 'Date' not in test_df.columns:
        test_df.rename(columns={'index': 'Date'}, inplace=True)

    # 2. Initialize Meta-Controller
    controller = MetaController()
    
    # 3. Setup Env
    # Note: Env expects numeric columns only, but having 'Date' as extra col is fine if we slice carefully
    env = RegimeAwareTradingEnv(test_df)
    obs, _ = env.reset()
    
    history = []
    done = False
    step_idx = 0
    
    while not done:
        # Get Data for current step
        row = test_df.iloc[step_idx]
        current_regime = int(row['Regime'])
        current_conf = row['Regime_Confidence']
        current_date = row['Date'] # Now we have the real date
        
        # Mock Probabilities [p0, p1, p2]
        probs = np.ones(config.N_REGIMES) * ((1 - current_conf) / (config.N_REGIMES - 1))
        probs[current_regime] = current_conf
        
        # Get Action
        final_action, agent_actions, scaler = controller.get_action(obs, probs, 0)
        
        # Step
        obs, reward, terminated, truncated, info = env.step(np.array([final_action]))
        done = terminated or truncated
        
        # Log
        history.append({
            'Date': current_date,
            'Net_Worth': info['net_worth'],
            'Regime': current_regime,
            'Confidence': current_conf,
            'Action': final_action,
            'Agent_0_Act': agent_actions[0],
            'Agent_1_Act': agent_actions[1],
            'Agent_2_Act': agent_actions[2]
        })
        step_idx += 1
        
    # Save Results
    results = pd.DataFrame(history)
    results.to_csv(os.path.join(config.PROJECT_ROOT, 'backtest_detailed_results.csv'), index=False)
    print("✅ Detailed results saved with correct dates.")

if __name__ == "__main__":
    run_backtest()