# src/training/test_env.py
import pandas as pd
import os
import sys
from stable_baselines3.common.env_checker import check_env

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import config
from trading_env import RegimeAwareTradingEnv

# Load Data
data_path = os.path.join(config.DATA_PROCESSED, 'market_data_with_regimes.csv')
df = pd.read_csv(data_path)

# Initialize Env
env = RegimeAwareTradingEnv(df)

# Check compliance with Gym API
print("🕵️ Checking Environment Compliance...")
check_env(env)
print("✅ Environment is Gymnasium Compliant!")

# Test a random loop
obs, _ = env.reset()
done = False
print("\n🎲 Running Random Simulation...")
while not done:
    action = env.action_space.sample() # Random action
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    if done:
        print(f"🏁 Episode finished. Final Net Worth: ${info['net_worth']:.2f}")