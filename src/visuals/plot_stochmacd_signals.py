"""
Stochastic MACD Strategy Visualization
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional


def plot_stochmacd_signals(data: pd.DataFrame, 
                       ticker: str,
                       strategy_params: dict,
                       start_date: Optional[str] = None, 
                       end_date: Optional[str] = None):
    """
    Plot Stochastic MACD strategy signals for a selected ticker.
    
    Args:
        data: DataFrame with indicators (from strategy.prepare_data())
        ticker: Ticker symbol to plot
        strategy_params: Dictionary with strategy parameters
        start_date: Start date (format: 'YYYY-MM-DD')
        end_date: End date (format: 'YYYY-MM-DD')
    """
    # Filter data for the selected ticker
    if 'ticker' in data.columns:
        ticker_data = data[data['ticker'] == ticker].copy()
    else:
        ticker_data = data.copy()
    
    if ticker_data.empty:
        print(f"❌ No data found for ticker: {ticker}")
        return
    
    # Apply date filter if provided
    if start_date:
        ticker_data = ticker_data[ticker_data.index >= start_date]
    if end_date:
        ticker_data = ticker_data[ticker_data.index <= end_date]
    
    if ticker_data.empty:
        print(f"❌ No data found for the selected period")
        return
    
    # Extract parameters
    period = strategy_params.get('period', 45)
    fast_period = strategy_params.get('fast_period', 12)
    slow_period = strategy_params.get('slow_period', 26)
    signal_period = strategy_params.get('signal_period', 9)
    overbought = strategy_params.get('overbought', 10)
    oversold = strategy_params.get('oversold', -10)
    use_trend_filter = strategy_params.get('use_trend_filter', False)
    use_rsi_filter = strategy_params.get('use_rsi_filter', False)
    
    # Print header
    print(f"\n" + "="*80)
    print(f"STOCHASTIC MACD STRATEGY VISUALIZATION FOR {ticker}")
    if start_date or end_date:
        date_range = f"{start_date or ticker_data.index[0].strftime('%Y-%m-%d')} to {end_date or ticker_data.index[-1].strftime('%Y-%m-%d')}"
        print(f"Period: {date_range}")
    print("="*80)
    
    # Check required columns
    required_cols = ['Close', 'STMACD', 'STMACD_SIGNAL', 'STMACD_HISTOGRAM']
    missing_cols = [col for col in required_cols if col not in ticker_data.columns]
    if missing_cols:
        print(f"⚠️  Warning: Missing columns: {missing_cols}")
        print("   Make sure to run prepare_data() first")
        return
    
    # Detect entry and exit signals
    entry_cond, exit_cond = _detect_stmacd_signals(ticker_data, strategy_params)
    entry_dates = ticker_data[entry_cond].index
    exit_dates = ticker_data[exit_cond].index
    
    # Create figure with subplots
    num_plots = 5 if use_rsi_filter else 4
    height_ratios = [3, 2, 1.5, 2, 2] if use_rsi_filter else [3, 2, 1.5, 2]
    
    fig, axes = plt.subplots(num_plots, 1, figsize=(16, 4*num_plots), 
                             gridspec_kw={'height_ratios': height_ratios})
    
    # ========================================================================
    # 1. Price Chart with Entry/Exit Signals
    # ========================================================================
    ax1 = axes[0]
    
    # Plot price
    ax1.plot(ticker_data.index, ticker_data['Close'], 
             label='Close Price', linewidth=1.8, color='black', alpha=0.8)
    
    # Plot trend filter if enabled
    if use_trend_filter and 'TREND_EMA' in ticker_data.columns:
        ax1.plot(ticker_data.index, ticker_data['TREND_EMA'], 
                label=f'Trend EMA({strategy_params.get("trend_ema_period", 50)})', 
                linewidth=1.5, color='blue', alpha=0.6, linestyle='--')
        
        # Shade bullish/bearish zones
        above_ema = ticker_data['Close'] > ticker_data['TREND_EMA']
        ax1.fill_between(ticker_data.index,
                        ticker_data['Close'].min(),
                        ticker_data['Close'].max(),
                        where=above_ema, alpha=0.05, color='green',
                        label='Bullish Zone')
    
    # Plot entry signals
    if len(entry_dates) > 0:
        ax1.scatter(entry_dates, 
                   ticker_data.loc[entry_dates, 'Close'],
                   color='darkgreen', s=150, marker='^', 
                   edgecolors='black', linewidth=2,
                   label='Entry (Bullish Cross)', zorder=5)
    
    # Plot exit signals
    if len(exit_dates) > 0:
        ax1.scatter(exit_dates, 
                   ticker_data.loc[exit_dates, 'Close'],
                   color='darkred', s=150, marker='v', 
                   edgecolors='black', linewidth=2,
                   label='Exit (Bearish Cross)', zorder=5)
    
    ax1.set_title(f'{ticker} - Stochastic MACD Strategy', 
                 fontsize=14, fontweight='bold')
    ax1.set_ylabel('Price ($)', fontsize=11)
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(alpha=0.3)
    
    # ========================================================================
    # 2. Stochastic MACD with Signal Line
    # ========================================================================
    ax2 = axes[1]
    
    # Plot STMACD and Signal lines
    ax2.plot(ticker_data.index, ticker_data['STMACD'], 
            label='STMACD', linewidth=1.8, color='blue', alpha=0.8)
    ax2.plot(ticker_data.index, ticker_data['STMACD_SIGNAL'], 
            label='Signal', linewidth=1.8, color='red', alpha=0.8)
    
    # Plot overbought/oversold levels
    ax2.axhline(y=overbought, color='red', linestyle='--', 
               alpha=0.6, linewidth=1.5, label=f'Overbought ({overbought})')
    ax2.axhline(y=oversold, color='green', linestyle='--', 
               alpha=0.6, linewidth=1.5, label=f'Oversold ({oversold})')
    ax2.axhline(y=0, color='gray', linestyle='-', 
               alpha=0.4, linewidth=1)
    
    # Shade overbought/oversold zones
    ax2.fill_between(ticker_data.index, overbought, 
                    ticker_data['STMACD'].max() * 1.1,
                    alpha=0.1, color='red', label='Overbought Zone')
    ax2.fill_between(ticker_data.index, oversold, 
                    ticker_data['STMACD'].min() * 1.1,
                    alpha=0.1, color='green', label='Oversold Zone')
    
    # Mark bullish crossovers
    for idx in entry_dates:
        if idx in ticker_data.index:
            ax2.scatter(idx, ticker_data.loc[idx, 'STMACD'],
                       color='darkgreen', s=100, marker='^',
                       edgecolors='black', linewidth=1.5, zorder=5)
    
    # Mark bearish crossovers
    for idx in exit_dates:
        if idx in ticker_data.index:
            ax2.scatter(idx, ticker_data.loc[idx, 'STMACD'],
                       color='darkred', s=100, marker='v',
                       edgecolors='black', linewidth=1.5, zorder=5)
    
    # Shade bullish/bearish regions
    stmacd_above = ticker_data['STMACD'] > ticker_data['STMACD_SIGNAL']
    ax2.fill_between(ticker_data.index,
                    ticker_data['STMACD'].min() * 1.1,
                    ticker_data['STMACD'].max() * 1.1,
                    where=stmacd_above, alpha=0.05, color='green')
    
    ax2.set_title('Stochastic MACD Oscillator', fontsize=12, fontweight='bold')
    ax2.set_ylabel('STMACD Value', fontsize=10)
    ax2.legend(loc='best', fontsize=9, ncol=2)
    ax2.grid(alpha=0.3)
    
    # ========================================================================
    # 3. STMACD Histogram
    # ========================================================================
    ax3 = axes[2]
    
    # Plot histogram
    colors = ['green' if h >= 0 else 'red' for h in ticker_data['STMACD_HISTOGRAM']]
    ax3.bar(ticker_data.index, ticker_data['STMACD_HISTOGRAM'],
           color=colors, alpha=0.6, width=0.8, label='STMACD Histogram')
    
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=1)
    
    # Mark crossover points
    for idx in entry_dates:
        if idx in ticker_data.index:
            ax3.axvline(x=idx, color='green', alpha=0.3, 
                       linestyle='--', linewidth=1.5)
    
    for idx in exit_dates:
        if idx in ticker_data.index:
            ax3.axvline(x=idx, color='red', alpha=0.3, 
                       linestyle='--', linewidth=1.5)
    
    ax3.set_title('STMACD Histogram (STMACD - Signal)', 
                 fontsize=12, fontweight='bold')
    ax3.set_ylabel('Histogram', fontsize=10)
    ax3.legend(loc='best', fontsize=9)
    ax3.grid(alpha=0.3)
    
    # ========================================================================
    # 4. RSI (if enabled)
    # ========================================================================
    if use_rsi_filter and 'RSI' in ticker_data.columns:
        ax4 = axes[3]
        
        # Plot RSI
        ax4.plot(ticker_data.index, ticker_data['RSI'], 
                label='RSI', linewidth=1.8, color='purple', alpha=0.8)
        
        # Plot threshold
        rsi_threshold = strategy_params.get('rsi_threshold', 50)
        ax4.axhline(y=rsi_threshold, color='blue', linestyle='--', 
                   alpha=0.6, linewidth=1.5,
                   label=f'Threshold ({rsi_threshold})')
        ax4.axhline(y=70, color='red', linestyle='--', 
                   alpha=0.4, linewidth=1, label='Overbought (70)')
        ax4.axhline(y=30, color='green', linestyle='--', 
                   alpha=0.4, linewidth=1, label='Oversold (30)')
        ax4.axhline(y=50, color='gray', linestyle='-', 
                   alpha=0.3, linewidth=1)
        
        # Shade zones
        ax4.fill_between(ticker_data.index, rsi_threshold, 100,
                        alpha=0.05, color='green', label='Bullish Zone')
        ax4.fill_between(ticker_data.index, 0, rsi_threshold,
                        alpha=0.05, color='red', label='Bearish Zone')
        
        ax4.set_title('RSI Momentum Filter', fontsize=12, fontweight='bold')
        ax4.set_ylabel('RSI', fontsize=10)
        ax4.set_ylim(0, 100)
        ax4.legend(loc='best', fontsize=9)
        ax4.grid(alpha=0.3)
        
        volume_ax_idx = 4
    else:
        volume_ax_idx = 3
    
    # ========================================================================
    # 5. Volume
    # ========================================================================
    ax_vol = axes[volume_ax_idx]
    
    if 'Volume' in ticker_data.columns:
        # Plot volume bars
        colors = ['green' if c >= o else 'red' 
                 for c, o in zip(ticker_data['Close'], ticker_data['Open'])]
        ax_vol.bar(ticker_data.index, ticker_data['Volume'], 
                  alpha=0.4, color=colors, width=0.8)
        
        # Volume moving average
        vol_ma = ticker_data['Volume'].rolling(window=20).mean()
        ax_vol.plot(ticker_data.index, vol_ma, 
                   color='blue', linewidth=1.5, 
                   label='Volume MA(20)', alpha=0.7)
        
        # Mark high volume on signals
        for idx in entry_dates:
            if idx in ticker_data.index:
                ax_vol.scatter(idx, ticker_data.loc[idx, 'Volume'],
                             color='green', s=80, marker='^',
                             edgecolors='darkgreen', linewidth=1, zorder=5)
        
        for idx in exit_dates:
            if idx in ticker_data.index:
                ax_vol.scatter(idx, ticker_data.loc[idx, 'Volume'],
                             color='red', s=80, marker='v',
                             edgecolors='darkred', linewidth=1, zorder=5)
    
    ax_vol.set_title('Volume', fontsize=12, fontweight='bold')
    ax_vol.set_ylabel('Volume', fontsize=10)
    ax_vol.set_xlabel('Date', fontsize=10)
    ax_vol.legend(loc='best', fontsize=9)
    ax_vol.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    

def _detect_stmacd_signals(data: pd.DataFrame, 
                          params: dict) -> tuple[pd.Series, pd.Series]:
    """
    Detect STMACD entry and exit signals.
    
    Returns:
        (entry_conditions, exit_conditions) as boolean Series
    """
    # Entry: Bullish crossover (STMACD crosses above Signal)
    bullish_cross = (
        (data['STMACD'] >= data['STMACD_SIGNAL']) &
        (data['STMACD'].shift(1) < data['STMACD_SIGNAL'].shift(1))
    )
    
    entry = bullish_cross
    
    # Optional threshold filter
    if params.get('use_threshold_filter', True):
        overbought = params.get('overbought', 10)
        not_overbought = data['STMACD'] < overbought
        entry = entry & not_overbought
    
    # Optional trend filter
    if params.get('use_trend_filter', False) and 'TREND_EMA' in data.columns:
        above_ema = data['Close'] > data['TREND_EMA']
        entry = entry & above_ema
    
    # Optional RSI filter
    if params.get('use_rsi_filter', False) and 'RSI' in data.columns:
        rsi_threshold = params.get('rsi_threshold', 50)
        rsi_bullish = data['RSI'] > rsi_threshold
        entry = entry & rsi_bullish
    
    # Exit: Bearish crossover (STMACD crosses below Signal)
    bearish_cross = (
        (data['STMACD'] < data['STMACD_SIGNAL']) &
        (data['STMACD'].shift(1) >= data['STMACD_SIGNAL'].shift(1))
    )
    
    exit_signal = bearish_cross
    
    # Optional early exit on overbought decline
    if params.get('use_threshold_filter', True):
        overbought = params.get('overbought', 10)
        overbought_decline = (
            (data['STMACD'].shift(1) >= overbought) &
            (data['STMACD'] < data['STMACD'].shift(1))
        )
        exit_signal = exit_signal | overbought_decline
    
    return entry, exit_signal

