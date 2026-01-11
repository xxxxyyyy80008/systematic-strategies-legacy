
"""
AMA-KAMA Strategy Visualization
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional


def plot_ama_kama_signals(data: pd.DataFrame, 
                          ticker: str,
                          strategy_params: dict,
                          start_date: Optional[str] = None, 
                          end_date: Optional[str] = None):
    """
    Plot AMA-KAMA strategy signals for a selected ticker.
    
    Args:
        data: DataFrame with ticker data and indicators
        ticker: Ticker symbol to plot
        strategy_params: Dictionary with strategy parameters (from YAML config)
        start_date: Start date (format: 'YYYY-MM-DD')
        end_date: End date (format: 'YYYY-MM-DD')
    """
    
    # Filter data for the selected ticker
    if 'ticker' in data.columns:
        ticker_data = data[data['ticker'] == ticker].copy()
    else:
        ticker_data = data.copy()
        ticker_data['ticker'] = ticker
    
    if ticker_data.empty:
        print(f" No data found for ticker: {ticker}")
        return
    
    # Apply date filter if provided
    if start_date:
        ticker_data = ticker_data[ticker_data.index >= start_date]
    if end_date:
        ticker_data = ticker_data[ticker_data.index <= end_date]
    
    if ticker_data.empty:
        print(f" No data found for the selected period")
        return
    
    # Extract strategy parameters
    ama_period = strategy_params.get('ama_period', 10)
    ama_fast = strategy_params.get('ama_fast', 2)
    ama_slow = strategy_params.get('ama_slow', 30)
    ema_period = strategy_params.get('ema_period', 20)
    rsi_period = strategy_params.get('rsi_period', 14)
    rsi_entry_max = strategy_params.get('rsi_entry_max', 35)
    rsi_exit_min = strategy_params.get('rsi_exit_min', 65)
    
    # Check required columns
    required_cols = ['Close', 'AMA', 'EMA', 'RSI']
    missing_cols = [col for col in required_cols if col not in ticker_data.columns]
    if missing_cols:
        print(f"  Warning: Missing columns: {missing_cols}")
        print("   Make sure to run prepare_data() first")
        return
    
    # Print header
    print(f"\n" + "="*80)
    print(f"AMA-KAMA STRATEGY VISUALIZATION FOR {ticker}")
    if start_date or end_date:
        date_range = f"{start_date or ticker_data.index[0].strftime('%Y-%m-%d')} to {end_date or ticker_data.index[-1].strftime('%Y-%m-%d')}"
        print(f"Period: {date_range}")
    print("="*80)
    
    # Extract parameters
    ama_period = strategy_params.get('ama_period', 10)
    ama_fast = strategy_params.get('ama_fast', 2)
    ama_slow = strategy_params.get('ama_slow', 30)
    ema_period = strategy_params.get('ema_period', 20)
    rsi_entry_max = strategy_params.get('rsi_entry_max', 35)
    rsi_exit_min = strategy_params.get('rsi_exit_min', 65)
    
    # Create figure with subplots
    fig, axes = plt.subplots(4, 1, figsize=(16, 14), 
                             gridspec_kw={'height_ratios': [3, 2, 2, 2]})
    
    # 1. Price with AMA and EMA
    ax1 = axes[0]
    
    # Plot price
    ax1.plot(ticker_data.index, ticker_data['Close'], 
             label='Close Price', linewidth=1.5, color='black', alpha=0.8)
    
    # Plot AMA
    if 'AMA' in ticker_data.columns:
        ax1.plot(ticker_data.index, ticker_data['AMA'], 
                label=f'AMA({ama_period},{ama_fast},{ama_slow})', 
                linewidth=2.0, color='blue', alpha=0.7)
    
    # Plot EMA
    if 'EMA' in ticker_data.columns:
        ax1.plot(ticker_data.index, ticker_data['EMA'], 
                label=f'EMA({ema_period})', 
                linewidth=2.0, color='red', alpha=0.7)
    
    # Mark entry conditions (AMA > EMA, RSI < entry_max)
    if all(col in ticker_data.columns for col in ['AMA', 'EMA', 'RSI']):
        entry_conditions = (
            (ticker_data['AMA'] > ticker_data['EMA']) &
            (ticker_data['AMA'].shift(1) <= ticker_data['EMA'].shift(1)) &
            (ticker_data['RSI'] < rsi_entry_max)
        )
        entry_dates = ticker_data[entry_conditions].index
        
        if len(entry_dates) > 0:
            ax1.scatter(entry_dates, ticker_data.loc[entry_dates, 'Close'],
                       color='green', s=100, marker='^', 
                       edgecolors='darkgreen', linewidth=1.5,
                       label='Entry Signal', zorder=5, alpha=0.7)
    
    # Mark exit conditions (AMA < EMA, RSI > exit_min)
    if all(col in ticker_data.columns for col in ['AMA', 'EMA', 'RSI']):
        exit_conditions = (
            (ticker_data['AMA'] < ticker_data['EMA']) &
            (ticker_data['AMA'].shift(1) >= ticker_data['EMA'].shift(1)) &
            (ticker_data['RSI'] > rsi_exit_min)
        )
        exit_dates = ticker_data[exit_conditions].index
        
        if len(exit_dates) > 0:
            ax1.scatter(exit_dates, ticker_data.loc[exit_dates, 'Close'],
                       color='red', s=100, marker='v', 
                       edgecolors='darkred', linewidth=1.5,
                       label='Exit Signal', zorder=5, alpha=0.7)
    
    # Highlight bullish/bearish regions
    if all(col in ticker_data.columns for col in ['AMA', 'EMA']):
        ama_above = ticker_data['AMA'] > ticker_data['EMA']
        ax1.fill_between(ticker_data.index, 
                        ticker_data['Close'].min(), 
                        ticker_data['Close'].max(),
                        where=ama_above, alpha=0.05, color='green',
                        label='Bullish Zone (AMA > EMA)')
    
    ax1.set_title(f'{ticker} - AMA-KAMA Strategy', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Price ($)', fontsize=11)
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(alpha=0.3)
    
    # 2. AMA-EMA Distance
    ax2 = axes[1]
    
    if all(col in ticker_data.columns for col in ['AMA', 'EMA']):
        # Calculate distance
        distance = ((ticker_data['AMA'] - ticker_data['EMA']) / ticker_data['EMA'] * 100)
        
        ax2.plot(ticker_data.index, distance, 
                label='AMA - EMA Distance (%)', linewidth=1.5, color='purple')
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # Fill positive/negative regions
        ax2.fill_between(ticker_data.index, 0, distance, 
                         where=(distance >= 0), 
                         alpha=0.3, color='green', label='AMA > EMA')
        ax2.fill_between(ticker_data.index, 0, distance, 
                         where=(distance < 0), 
                         alpha=0.3, color='red', label='EMA > AMA')
    
    ax2.set_title('AMA-EMA Distance', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Distance (%)', fontsize=10)
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(alpha=0.3)
    
    # 3. RSI with Entry/Exit Levels
    ax3 = axes[2]
    
    if 'RSI' in ticker_data.columns:
        ax3.plot(ticker_data.index, ticker_data['RSI'], 
                label='RSI(14)', linewidth=1.5, color='purple')
        
        # Reference lines
        ax3.axhline(y=rsi_entry_max, color='green', linestyle='--', 
                   alpha=0.6, linewidth=1.5,
                   label=f'Entry Max ({rsi_entry_max})')
        ax3.axhline(y=rsi_exit_min, color='red', linestyle='--', 
                   alpha=0.6, linewidth=1.5,
                   label=f'Exit Min ({rsi_exit_min})')
        ax3.axhline(y=50, color='gray', linestyle='-', 
                   alpha=0.3, linewidth=1)
        ax3.axhline(y=70, color='orange', linestyle='--', 
                   alpha=0.4, linewidth=1, label='Overbought (70)')
        ax3.axhline(y=30, color='lightgreen', linestyle='--', 
                   alpha=0.4, linewidth=1, label='Oversold (30)')
        
        # Fill zones
        ax3.fill_between(ticker_data.index, 0, rsi_entry_max, 
                         alpha=0.1, color='green', label='')
        ax3.fill_between(ticker_data.index, rsi_exit_min, 100, 
                         alpha=0.1, color='red', label='')
    
    ax3.set_title('Relative Strength Index (RSI)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('RSI', fontsize=10)
    ax3.set_ylim(0, 100)
    ax3.legend(loc='best', fontsize=9)
    ax3.grid(alpha=0.3)
    
    # 4. Volume (if available)
    ax4 = axes[3]
    
    if 'Volume' in ticker_data.columns:
        # Plot volume bars
        if 'Open' in ticker_data.columns:
            colors = ['green' if close >= open_price else 'red' 
                     for close, open_price in zip(ticker_data['Close'], 
                                                   ticker_data['Open'])]
        else:
            colors = ['gray'] * len(ticker_data)
        
        ax4.bar(ticker_data.index, ticker_data['Volume'], 
               alpha=0.4, color=colors, label='Volume', width=0.8)
        
        # Add volume moving average
        vol_ma = ticker_data['Volume'].rolling(window=20).mean()
        ax4.plot(ticker_data.index, vol_ma, 
                label='Volume MA(20)', 
                linewidth=1.5, color='blue', alpha=0.7)
    
    ax4.set_title('Volume', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Volume', fontsize=10)
    ax4.set_xlabel('Date', fontsize=10)
    ax4.legend(loc='best', fontsize=9)
    ax4.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
