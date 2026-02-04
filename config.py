# config.py
import os

# --- PATHS ---
# Automatically detect project root
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_RAW = os.path.join(PROJECT_ROOT, 'data', 'raw')
DATA_PROCESSED = os.path.join(PROJECT_ROOT, 'data', 'processed')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'saved_agents')
LOGS_DIR = os.path.join(PROJECT_ROOT, 'logs')

# --- DATA SETTINGS ---
# Real assets. SPY (Market), VIX (Volatility), GLD (Safe Haven), AAPL (Tech)
TICKERS = ["SPY", "^VIX", "GLD", "AAPL"] 
START_DATE = "2010-01-01"
END_DATE = "2023-12-31" # Fixed end date for reproducibility
INTERVAL = "1d"         # Daily data for macro strategies (Use '1h' for crypto)

# --- FEATURE ENGINEERING ---
# The lookback window for the model to "see"
WINDOW_SIZE = 30 
# Features to exclude from normalization (e.g., categorical or sin/cos time)
NON_STATIONARY_COLS = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']

# --- REGIME DETECTION ---
# Number of regimes: 0=Bear/Crash, 1=Sideways, 2=Bull/Trend
N_REGIMES = 3 

# --- RL HYPERPARAMETERS ---
TOTAL_TIMESTEPS = 100_000
LEARNING_RATE = 0.0003
GAMMA = 0.95            # Discount factor
ENTROPY_BETA = 0.05     # Exploration incentive

# Ensure directories exist
for d in [DATA_RAW, DATA_PROCESSED, MODELS_DIR, LOGS_DIR]:
    os.makedirs(d, exist_ok=True)