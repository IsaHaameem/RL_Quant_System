# src/data_loader/downloader.py
import yfinance as yf
import pandas as pd
import os
import sys

# Add project root to path to import config
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import config

class MarketDataDownloader:
    def __init__(self, tickers=config.TICKERS, start=config.START_DATE, end=config.END_DATE):
        self.tickers = tickers
        self.start = start
        self.end = end
        
    def download_data(self):
        """
        Downloads data for all tickers and merges them into a single DataFrame.
        Format: Multi-index or Prefix columns (SPY_Close, VIX_Close).
        """
        print(f"⬇️ Downloading data for: {self.tickers} from {self.start} to {self.end}...")
        
        # Download all at once (more efficient)
        raw_data = yf.download(
            self.tickers, 
            start=self.start, 
            end=self.end, 
            interval=config.INTERVAL,
            group_by='ticker',
            auto_adjust=True  # Adjusts for splits/dividends automatically
        )
        
        if raw_data.empty:
            raise ValueError("❌ No data downloaded. Check your internet or ticker symbols.")

        # Flatten the MultiIndex columns if necessary
        # yfinance returns (Ticker, OHLCV). We want flat columns: Ticker_Feature
        df_flat = pd.DataFrame()
        
        for ticker in self.tickers:
            # Handle case where only 1 ticker is downloaded (structure differs)
            if len(self.tickers) == 1:
                ticker_df = raw_data.copy()
            else:
                try:
                    ticker_df = raw_data[ticker].copy()
                except KeyError:
                    print(f"⚠️ Warning: {ticker} data not found in response.")
                    continue
            
            # Rename columns to {Ticker}_{Feature}
            ticker_df.columns = [f"{ticker}_{col}" for col in ticker_df.columns]
            
            if df_flat.empty:
                df_flat = ticker_df
            else:
                df_flat = df_flat.join(ticker_df, how='outer')

        # Forward fill missing data (essential for trading alignment)
        df_flat.ffill(inplace=True)
        df_flat.dropna(inplace=True) # Drop initial rows that have NaNs

        save_path = os.path.join(config.DATA_RAW, 'market_data.csv')
        df_flat.to_csv(save_path)
        print(f"✅ Data saved successfully to: {save_path}")
        print(f"   Shape: {df_flat.shape}")
        return df_flat

if __name__ == "__main__":
    # Test run
    downloader = MarketDataDownloader()
    downloader.download_data()