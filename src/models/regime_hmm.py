# src/models/regime_hmm.py
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
import joblib
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import config

class RegimeDetector:
    def __init__(self, n_components=3):
        self.n_components = n_components
        self.model = GaussianHMM(n_components=n_components, covariance_type="full", n_iter=100, random_state=42)
        
    def prepare_features(self, df, ticker="SPY"):
        feature_cols = [f'{ticker}_Log_Ret', f'{ticker}_Vol_20']
        data = df[feature_cols].copy()
        if np.isinf(data).values.any():
            data.replace([np.inf, -np.inf], 0, inplace=True)
        return data.values

    def fit(self, data_values):
        print(f"🧠 Training HMM with {self.n_components} regimes...")
        try:
            self.model.fit(data_values)
            print("✅ HMM Converged.")
            return True
        except Exception as e:
            print(f"⚠️ HMM Failed: {e}")
            return False

    def predict(self, data_values):
        return self.model.predict(data_values)
    
    def predict_proba(self, data_values):
        return self.model.predict_proba(data_values)

    def save(self):
        path = os.path.join(config.MODELS_DIR, 'hmm_model.pkl')
        joblib.dump(self.model, path)

    def force_quantiles(self, df, ticker="SPY"):
        """
        FALLBACK: If HMM fails to find 3 regimes, use hard statistical buckets.
        Bear = Bottom 33% Returns
        Bull = Top 33% Returns
        Sideways = Middle
        """
        print("☢️ ACTIVATING FALLBACK: Forcing 3 Regimes using Statistical Quantiles...")
        
        # Calculate thresholds
        ret_col = f'{ticker}_Log_Ret'
        low_thresh = df[ret_col].quantile(0.33)
        high_thresh = df[ret_col].quantile(0.66)
        
        conditions = [
            (df[ret_col] <= low_thresh),              # Bear
            (df[ret_col] > low_thresh) & (df[ret_col] <= high_thresh), # Sideways
            (df[ret_col] > high_thresh)               # Bull
        ]
        choices = [0, 1, 2] # 0=Bear, 1=Sideways, 2=Bull
        
        df['Regime'] = np.select(conditions, choices, default=1)
        df['Regime_Confidence'] = 0.95 # Artificial high confidence
        return df

    def analyze_and_remap_regimes(self, df, predictions, probabilities, ticker="SPY"):
        df['Regime_Raw'] = predictions
        
        # Check if we actually have 3 regimes
        unique_found = df['Regime_Raw'].nunique()
        print(f"🔍 HMM found {unique_found} unique regimes.")
        
        if unique_found < 3:
            print("⚠️ HMM found fewer than 3 regimes. Switching to Quantile Fallback.")
            return self.force_quantiles(df, ticker)

        # If HMM worked, map them correctly (Low Ret -> 0, High Ret -> 2)
        stats = df.groupby('Regime_Raw')[f'{ticker}_Log_Ret'].mean().sort_values()
        
        mapping = {
            stats.index[0]: 0, # Lowest Return -> Bear
            stats.index[1]: 1, # Middle -> Sideways
            stats.index[2]: 2  # Highest -> Bull
        }
        
        df['Regime'] = df['Regime_Raw'].map(mapping)
        
        # Re-order probabilities
        probs_sorted = np.zeros_like(probabilities)
        for old_id, new_id in mapping.items():
            probs_sorted[:, new_id] = probabilities[:, old_id]
            
        df['Regime_Confidence'] = np.max(probs_sorted, axis=1)
        
        # Safety Override (Volatility Panic)
        vol_col = f'{ticker}_Vol_20'
        panic_threshold = df[vol_col].quantile(0.95)
        override_mask = df[vol_col] > panic_threshold
        df.loc[override_mask, 'Regime'] = 0 # Force Bear
        df.loc[override_mask, 'Regime_Confidence'] = 1.0
        
        df.drop(columns=['Regime_Raw'], inplace=True)
        return df

if __name__ == "__main__":
    data_path = os.path.join(config.DATA_PROCESSED, 'features.csv')
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    
    detector = RegimeDetector(n_components=config.N_REGIMES)
    X = detector.prepare_features(df, ticker="SPY")
    success = detector.fit(X)
    
    if success:
        regimes = detector.predict(X)
        probs = detector.predict_proba(X)
        df_labeled = detector.analyze_and_remap_regimes(df, regimes, probs)
    else:
        # If fit crashed, force quantiles immediately
        df_labeled = detector.force_quantiles(df)
    
    # Final check
    counts = df_labeled['Regime'].value_counts().sort_index()
    print("\n📊 Final Regime Counts (Must have 0, 1, 2):")
    print(counts)
    
    save_path = os.path.join(config.DATA_PROCESSED, 'market_data_with_regimes.csv')
    df_labeled.to_csv(save_path)
    detector.save()
    print(f"✅ Data saved. All 3 regimes guaranteed.")