"""
data downloader for Yahoo Finance.

"""

import pandas as pd
from typing import Dict, List, Optional, Tuple
from functools import wraps
import warnings

warnings.filterwarnings('ignore')



def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize DataFrame column names to title case.
    
    Args:
        df: Raw DataFrame from data source
        
    Returns:
        DataFrame with standardized column names
    """
    column_mapping = {
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume',
        'adj close': 'Adj Close',
        'adjusted_close': 'Adj Close',
        'dividends': 'Dividends',
        'stock splits': 'Stock Splits'
    }
    
    df_clean = df.copy()
    df_clean.columns = [column_mapping.get(col.lower(), col) for col in df_clean.columns]
    
    return df_clean


def validate_dataframe(df: pd.DataFrame) -> bool:
    """
    Check if DataFrame has required OHLCV columns and valid data.
    
    Args:
        df: DataFrame to validate
        
    Returns:
        True if valid, False otherwise
    """
    if df.empty:
        return False
    
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    return all(col in df.columns for col in required_cols)


def clean_ticker(ticker: str) -> str:
    """
    Clean and standardize ticker symbol.
    
    Args:
        ticker: Raw ticker symbol
        
    Returns:
        Cleaned ticker symbol
    """
    return ticker.strip().upper()


def retry_on_failure(max_attempts: int = 3):
    """
    Decorator to retry function on failure.
    
    Args:
        max_attempts: Maximum number of retry attempts
        
    Returns:
        Decorated function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise
                    continue
        return wrapper
    return decorator


@retry_on_failure(max_attempts=3)
def download_single_ticker(ticker: str, 
                          start_date: str,
                          end_date: Optional[str] = None,
                          interval: str = '1d') -> Optional[pd.DataFrame]:
    """
    Download data for a single ticker from Yahoo Finance.
    
    Args:
        ticker: Stock ticker symbol
        start_date: Start date (format: 'YYYY-MM-DD')
        end_date: End date (format: 'YYYY-MM-DD'), None for current date
        interval: Data interval ('1d', '1wk', '1mo')
        
    Returns:
        DataFrame with OHLCV data, or None if download fails
    """
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("yfinance not installed. Run: pip install yfinance")
    
    try:
        ticker_obj = yf.Ticker(ticker)
        
        if end_date:
            hist = ticker_obj.history(start=start_date, end=end_date, interval=interval)
        else:
            hist = ticker_obj.history(start=start_date, interval=interval)
        
        if hist.empty:
            return None
        
        # Standardize columns
        hist = standardize_columns(hist)
        
        # Validate
        if not validate_dataframe(hist):
            return None
        
        return hist
    
    except Exception as e:
        print(f"Error downloading {ticker}: {str(e)}")
        return None


def download_data(tickers: List[str],
                 start_date: str,
                 end_date: Optional[str] = None,
                 interval: str = '1d',
                 verbose: bool = True) -> Dict[str, pd.DataFrame]:
    """
    Download data for multiple tickers from Yahoo Finance.
    
    Args:
        tickers: List of ticker symbols
        start_date: Start date (format: 'YYYY-MM-DD')
        end_date: End date (format: 'YYYY-MM-DD'), None for current date
        interval: Data interval ('1d', '1wk', '1mo')
        verbose: Print progress messages
        
    Returns:
        Dictionary mapping ticker to DataFrame with OHLCV data
        
    Example:
        >>> data = download_data(['AAPL', 'MSFT'], '2020-01-01')
        >>> aapl_df = data['AAPL']
        >>> print(aapl_df.head())
    """
    # Clean tickers
    clean_tickers = [clean_ticker(t) for t in tickers]
    
    if verbose:
        print("Downloading data...\n")
    
    results = {}
    failed = []
    
    for ticker in clean_tickers:
        if verbose:
            print(f"{ticker}...", end=' ')
        
        df = download_single_ticker(ticker, start_date, end_date, interval)
        
        if df is not None:
            results[ticker] = df
            if verbose:
                print(f"{len(df)} bars")
        else:
            failed.append(ticker)
            if verbose:
                print("Failed")
    
    if verbose:
        print(f"\nSuccessfully downloaded {len(results)}/{len(clean_tickers)} tickers")
        if failed:
            print(f"Failed: {', '.join(failed)}")
        print()
    
    return results


# UTILITY FUNCTIONS

def get_date_range(data: Dict[str, pd.DataFrame]) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """
    Get common date range across all tickers.
    
    Args:
        data: Dictionary of ticker DataFrames
        
    Returns:
        Tuple of (start_date, end_date)
    """
    if not data:
        return None, None
    
    all_dates = [df.index for df in data.values()]
    common_dates = set.intersection(*[set(dates) for dates in all_dates])
    
    if not common_dates:
        return None, None
    
    return min(common_dates), max(common_dates)


def filter_common_dates(data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Filter all DataFrames to only include common dates.
    
    Args:
        data: Dictionary of ticker DataFrames
        
    Returns:
        Dictionary with filtered DataFrames
    """
    if not data:
        return {}
    
    # Find common dates
    all_dates = [set(df.index) for df in data.values()]
    common_dates = set.intersection(*all_dates)
    
    if not common_dates:
        return data
    
    # Filter each DataFrame
    return {
        ticker: df.loc[df.index.isin(common_dates)].sort_index()
        for ticker, df in data.items()
    }


def combine_data(data: Dict[str, pd.DataFrame], 
                column: str = 'Close') -> pd.DataFrame:
    """
    Combine single column from multiple tickers into one DataFrame.
    
    Args:
        data: Dictionary of ticker DataFrames
        column: Column name to extract
        
    Returns:
        DataFrame with tickers as columns
        
    Example:
        >>> data = download_data(['AAPL', 'MSFT'], '2020-01-01')
        >>> prices = combine_data(data, 'Close')
        >>> print(prices.head())
                        AAPL    MSFT
        2020-01-02    75.09  160.62
        2020-01-03    74.36  158.62
    """
    if not data:
        return pd.DataFrame()
    
    combined = pd.DataFrame({
        ticker: df[column] 
        for ticker, df in data.items() 
        if column in df.columns
    })
    
    return combined.sort_index()


def export_to_csv(data: Dict[str, pd.DataFrame], output_dir: str = 'data') -> List[str]:
    """
    Export each ticker's data to separate CSV file.
    
    Args:
        data: Dictionary of ticker DataFrames
        output_dir: Output directory path
        
    Returns:
        List of created file paths
    """
    import os
    
    os.makedirs(output_dir, exist_ok=True)
    
    created_files = []
    
    for ticker, df in data.items():
        filepath = os.path.join(output_dir, f"{ticker}.csv")
        df.to_csv(filepath)
        created_files.append(filepath)
        print(f"Saved {ticker} to {filepath}")
    
    return created_files


