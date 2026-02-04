# src/training/trading_env.py
import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
import sys
import os

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import config

class RegimeAwareTradingEnv(gym.Env):
    """
    A custom Trading Environment that includes Regime Signals and Macro Data.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, df, initial_balance=10_000, commission=0.001):
        super(RegimeAwareTradingEnv, self).__init__()
        
        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.commission = commission
        
        # --- ACTION SPACE ---
        # Continuous: -1.0 (Full Short) to +1.0 (Full Long)
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)

        # --- OBSERVATION SPACE ---
        # 1. Market Returns (1)
        # 2. Market Volatility (1)
        # 3. Macro/VIX (1) if available
        # 4. Regime Signal (One-Hot Encoded for 3 regimes -> 3 columns)
        # 5. Position State (Current Holdings) (1)
        # Total Features = 1 + 1 + 1 + 3 + 1 = 7 features
        
        # We need to determine exact feature count dynamically
        self.feature_cols = [c for c in df.columns if 'Log_Ret' in c or 'Vol_20' in c or 'VIX' in c]
        self.n_features = len(self.feature_cols) + config.N_REGIMES + 1 # +1 for current position
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.n_features,), dtype=np.float32
        )

        # State Variables
        self.current_step = 0
        self.balance = initial_balance
        self.shares_held = 0
        self.net_worth = initial_balance
        self.max_net_worth = initial_balance
        self.current_position = 0.0 # -1 to 1 representation

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        self.current_position = 0.0
        
        return self._next_observation(), {}

    def _next_observation(self):
        # Get data for current step
        frame = self.df.iloc[self.current_step]
        
        # Extract features
        market_features = frame[self.feature_cols].values.astype(np.float32)
        
        # One-Hot Encode Regime
        regime = int(frame['Regime'])
        regime_one_hot = np.zeros(config.N_REGIMES, dtype=np.float32)
        regime_one_hot[regime] = 1.0
        
        # Combine all features
        # CRITICAL FIX: Ensure the list inputs are explicitly float32
        obs = np.concatenate([
            market_features, 
            regime_one_hot, 
            np.array([self.current_position], dtype=np.float32) 
        ])
        
        # DOUBLE CHECK: Force final cast to float32
        return obs.astype(np.float32)

    def step(self, action):
        # 1. Execute Trade
        self.current_step += 1
        
        # Scale action from [-1, 1] to actual portfolio weight
        target_weight = np.clip(action[0], -1, 1)
        
        # Rebalance portfolio logic
        weight_change = abs(target_weight - self.current_position)
        cost = weight_change * self.net_worth * self.commission
        
        # Update Net Worth
        log_ret = self.df.iloc[self.current_step]['SPY_Log_Ret']
        market_return = np.exp(log_ret) - 1
        
        portfolio_return = (self.current_position * market_return)
        self.net_worth = self.net_worth * (1 + portfolio_return) - cost
        self.current_position = target_weight
        
        # 2. Calculate Reward
        step_return = (self.net_worth - self.initial_balance) / self.initial_balance # Simple ROI for now
        
        # Reward = Change in Net Worth (scaled)
        # We use a simple PnL reward for stability first, can be enhanced later
        reward = portfolio_return * 100 
        
        # Penalty for Drawdown
        if self.net_worth < self.max_net_worth:
            drawdown = (self.max_net_worth - self.net_worth) / self.max_net_worth
            reward -= (drawdown * 0.01) 
            
        self.max_net_worth = max(self.net_worth, self.max_net_worth)
        
        # 3. Check Done (Strict Boolean Casting)
        # CRITICAL FIX: Wrap in bool()
        terminated = bool(self.net_worth <= (self.initial_balance * 0.5)) 
        truncated = bool(self.current_step >= len(self.df) - 2)
        
        info = {
            'net_worth': float(self.net_worth), # Ensure float
            'regime': int(self.df.iloc[self.current_step]['Regime']) # Ensure int
        }
        
        return self._next_observation(), reward, terminated, truncated, info