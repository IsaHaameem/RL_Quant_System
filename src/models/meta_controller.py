# src/models/meta_controller.py
import numpy as np
import os
import sys
from stable_baselines3 import PPO

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import config

class MetaController:
    def __init__(self):
        self.agents = {}
        self.load_agents()
        
    def load_agents(self):
        """Loads the 3 specialist agents into memory."""
        specialists_dir = os.path.join(config.MODELS_DIR, 'specialists')
        for i in range(config.N_REGIMES):
            path = os.path.join(specialists_dir, f'agent_regime_{i}.zip')
            if os.path.exists(path):
                print(f"📡 Loading Specialist {i}...")
                self.agents[i] = PPO.load(path)
            else:
                print(f"⚠️ Warning: Specialist {i} not found at {path} (Will default to 0.0 action)")

    def get_action(self, obs, regime_probs, current_volatility):
        """
        Returns the weighted action based on regime probabilities.
        """
        # Reshape obs for SB3
        obs_reshaped = obs.reshape(1, -1) if len(obs.shape) == 1 else obs
        
        actions = []
        for i in range(config.N_REGIMES):
            if i in self.agents:
                # Deterministic=True for stable backtesting
                act_array, _ = self.agents[i].predict(obs_reshaped, deterministic=True)
                # CRITICAL FIX: Extract the scalar float value
                # act_array is usually [0.5] -> we want 0.5
                action_value = float(act_array.item())
                actions.append(action_value)
            else:
                # Default to Neutral (0.0) if agent is missing
                actions.append(0.0)
                
        # 2. Compute Weighted Average Action
        # Now 'actions' is a list of floats, e.g., [-0.5, 0.8, 0.0]
        # regime_probs is array, e.g., [0.1, 0.8, 0.1]
        weighted_action = np.sum([a * p for a, p in zip(actions, regime_probs)])
        
        # 3. Confidence & Volatility Scaling
        confidence = np.max(regime_probs)
        size_scalar = 1.0
        
        # Simple Confidence Logic: If unsure (<60%), reduce size
        if confidence < 0.6:
            size_scalar = confidence 
            
        final_position = weighted_action * size_scalar
        
        return final_position, actions, size_scalar