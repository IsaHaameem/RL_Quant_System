# src/training/train.py
import pandas as pd
import os
import sys
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import config
from src.training.trading_env import RegimeAwareTradingEnv

def train_agent():
    print("🚀 Starting Training Pipeline...")
    
    # 1. Load Data
    data_path = os.path.join(config.DATA_PROCESSED, 'market_data_with_regimes.csv')
    df = pd.read_csv(data_path)
    
    # 2. Split Data (Train vs Test)
    # We want to train on older data and test on unseen recent data
    split_index = int(len(df) * 0.8)
    train_df = df.iloc[:split_index]
    test_df = df.iloc[split_index:]
    
    print(f"📊 Training Data: {len(train_df)} days")
    print(f"🧪 Testing Data: {len(test_df)} days")
    
    # 3. Create Environments
    # Stable-Baselines requires vectorized environments (even if just 1)
    train_env = DummyVecEnv([lambda: RegimeAwareTradingEnv(train_df)])
    test_env = DummyVecEnv([lambda: RegimeAwareTradingEnv(test_df)])
    
    # 4. Define the PPO Agent
    # MlpPolicy = Multi-Layer Perceptron (Standard Deep Neural Net)
    model = PPO(
        "MlpPolicy", 
        train_env, 
        verbose=1, 
        learning_rate=config.LEARNING_RATE,
        gamma=config.GAMMA,
        ent_coef=config.ENTROPY_BETA,
        tensorboard_log=config.LOGS_DIR,
        device="cuda" # Will auto-fallback to cpu if no GPU
    )
    
    # 5. Setup Evaluation Callback
    # This checks the agent's performance on the Test Set every 5000 steps
    # and saves the best version.
    eval_callback = EvalCallback(
        test_env,
        best_model_save_path=config.MODELS_DIR,
        log_path=config.LOGS_DIR,
        eval_freq=5000,
        deterministic=True,
        render=False
    )
    
    # 6. Train
    print("\n🧠 Agent is learning... (This may take a few minutes)")
    model.learn(total_timesteps=config.TOTAL_TIMESTEPS, callback=eval_callback)
    
    # 7. Final Save
    final_path = os.path.join(config.MODELS_DIR, 'final_agent.zip')
    model.save(final_path)
    print(f"✅ Training Complete. Model saved to {final_path}")

if __name__ == "__main__":
    train_agent()