"""Data downloading and preprocessing (pure functions)."""
from typing import Dict, List, Optional, Set
import pandas as pd
# import yfinance as yf


def download_ticker(ticker: str, start_date: str, 
                   end_date: Optional[str] = None) -> Optional[pd.DataFrame]:
    """
    Download data for single ticker (pure I/O function).
    
    Args:
        ticker: Ticker symbol
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (optional)
        
    Returns:
        DataFrame or None
    """
    try:
        data = yf.Ticker(ticker)
        hist = data.history(period="max", start=start_date, end=end_date)
        
        if hist.empty:
            return None
        
        # Handle multi-level columns from newer yfinance
        if isinstance(hist.columns, pd.MultiIndex):
            hist = hist.droplevel(1, axis=1)
        
        required = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in hist.columns for col in required):
            return None
        
        # Normalize timezone
        if hasattr(hist.index, 'tz') and hist.index.tz is not None:
            hist.index = hist.index.tz_convert("America/New_York")
            hist.index = pd.to_datetime(hist.index.date)
        
        return hist
    
    except Exception as e:
        print(f"Error downloading {ticker}: {str(e)}")
        return None


def download_multiple(tickers: List[str], start_date: str,
                     end_date: Optional[str] = None) -> Dict[str, pd.DataFrame]:
    """Download data for multiple tickers."""
    results = {}
    print(f"\nDownloading data for {len(tickers)} tickers...")
    
    for ticker in tickers:
        print(f"  {ticker}...", end=' ')
        df = download_ticker(ticker, start_date, end_date)
        
        if df is not None:
            results[ticker] = df
            print(f"✓ {len(df)} bars")
        else:
            print("✗ Failed")
    
    print(f"Successfully downloaded {len(results)}/{len(tickers)} tickers\n")
    return results


def find_common_dates(dataframes: List[pd.DataFrame]) -> Set:
    """Find common dates across DataFrames (pure function)."""
    if not dataframes:
        return set()
    
    date_sets = [set(df.index) for df in dataframes]
    return set.intersection(*date_sets)


def align_data(data_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Align all DataFrames to common dates (pure function).
    
    Returns new dict with aligned data.
    """
    if not data_dict:
        return {}
    
    common_dates = find_common_dates(list(data_dict.values()))
    
    if not common_dates:
        raise ValueError("No common dates found")
    
    return {
        ticker: df[df.index.isin(common_dates)].copy()
        for ticker, df in data_dict.items()
    }


def validate_ohlcv(df: pd.DataFrame) -> bool:
    """Validate OHLCV data (pure function)."""
    required = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    if not all(col in df.columns for col in required):
        return False
    
    if (df['High'] < df['Low']).any():
        return False
    
    if (df['Close'] <= 0).any():
        return False
    
    return True
