"""
Bollinger Bands Strategy Visualization
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional


def plot_bull_engulf_bb_signals(data: pd.DataFrame, 
                                ticker: str,
                                strategy_params: dict,
                                start_date: Optional[str] = None, 
                                end_date: Optional[str] = None):
    """
    Plot Bullish Engulfing + Bollinger Bands strategy signals.
    
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
    
    # Extract parameters
    bb_period = strategy_params.get('bb_period', 20)
    bb_std_dev = strategy_params.get('bb_std_dev', 2.0)
    require_engulfing = strategy_params.get('require_engulfing', True)
    use_rsi_filter = strategy_params.get('use_rsi_filter', False)
    rsi_oversold = strategy_params.get('rsi_oversold', 30)
    
    # Print header
    print(f"\n" + "="*80)
    print(f"BOLLINGER BANDS BOUNCE STRATEGY FOR {ticker}")
    if start_date or end_date:
        date_range = f"{start_date or ticker_data.index[0].strftime('%Y-%m-%d')} to {end_date or ticker_data.index[-1].strftime('%Y-%m-%d')}"
        print(f"Period: {date_range}")
    print("="*80)
    
    # Check required columns
    required_cols = ['Close', 'BB_UPPER', 'BB_MIDDLE', 'BB_LOWER', 
                     'BULL_ENGULF', 'BBANDS_BOUNCE', 'BBANDS_BREAKOUT']
    missing_cols = [col for col in required_cols if col not in ticker_data.columns]
    if missing_cols:
        print(f"  Warning: Missing columns: {missing_cols}")
        print("   Make sure to run prepare_data() first")
        return
    
    # Calculate additional metrics for visualization
    ticker_data = _add_visualization_metrics(ticker_data)
    
    # Create figure with subplots
    fig, axes = plt.subplots(4, 1, figsize=(16, 14), 
                             gridspec_kw={'height_ratios': [3, 2, 2, 2]})
    
    # ========================================================================
    # 1. Price Chart with Bollinger Bands
    # ========================================================================
    ax1 = axes[0]
    
    # Plot Bollinger Bands
    ax1.plot(ticker_data.index, ticker_data['BB_UPPER'], 
             label=f'BB Upper ({bb_period}, {bb_std_dev}σ)', 
             linewidth=1.5, color='red', alpha=0.6, linestyle='--')
    ax1.plot(ticker_data.index, ticker_data['BB_MIDDLE'], 
             label='BB Middle (SMA)', 
             linewidth=1.5, color='blue', alpha=0.6, linestyle='--')
    ax1.plot(ticker_data.index, ticker_data['BB_LOWER'], 
             label='BB Lower', 
             linewidth=1.5, color='green', alpha=0.6, linestyle='--')
    
    # Fill between bands
    ax1.fill_between(ticker_data.index, 
                     ticker_data['BB_UPPER'], 
                     ticker_data['BB_LOWER'],
                     alpha=0.08, color='gray', label='BB Band Width')
    
    # Plot price
    ax1.plot(ticker_data.index, ticker_data['Close'], 
             label='Close Price', linewidth=1.8, color='black', alpha=0.8)
    
    # Highlight Bullish Engulfing patterns
    bull_engulf_dates = ticker_data[ticker_data['BULL_ENGULF']].index
    if len(bull_engulf_dates) > 0:
        ax1.scatter(bull_engulf_dates, 
                   ticker_data.loc[bull_engulf_dates, 'Close'],
                   color='lightgreen', s=80, marker='o', 
                   alpha=0.5, label='Bullish Engulfing',
                   edgecolors='green', linewidth=1.5, zorder=4)
    
    # Highlight BBands Bounce signals
    bounce_dates = ticker_data[ticker_data['BBANDS_BOUNCE']].index
    if len(bounce_dates) > 0:
        ax1.scatter(bounce_dates, 
                   ticker_data.loc[bounce_dates, 'Close'],
                   color='orange', s=80, marker='s', 
                   alpha=0.6, label='BBands Bounce',
                   edgecolors='darkorange', linewidth=1.5, zorder=4)
    
    # Entry signals (Bullish Engulfing + BBands Bounce)
    if require_engulfing:
        entry_cond = ticker_data['BULL_ENGULF'] & ticker_data['BBANDS_BOUNCE']
    else:
        entry_cond = ticker_data['BBANDS_BOUNCE']
    
    if use_rsi_filter and 'RSI' in ticker_data.columns:
        entry_cond = entry_cond & (ticker_data['RSI'] < rsi_oversold)
    
    entry_dates = ticker_data[entry_cond].index
    if len(entry_dates) > 0:
        ax1.scatter(entry_dates, 
                   ticker_data.loc[entry_dates, 'Close'],
                   color='darkgreen', s=150, marker='^', 
                   edgecolors='black', linewidth=2,
                   label='Entry Signal', zorder=5)
    
    # Exit signals (BBands Breakout)
    exit_dates = ticker_data[ticker_data['BBANDS_BREAKOUT']].index
    if len(exit_dates) > 0:
        ax1.scatter(exit_dates, 
                   ticker_data.loc[exit_dates, 'Close'],
                   color='darkred', s=150, marker='v', 
                   edgecolors='black', linewidth=2,
                   label='Exit Signal (Breakout)', zorder=5)
    
    ax1.set_title(f'{ticker} - Bollinger Bands Bounce Strategy', 
                 fontsize=14, fontweight='bold')
    ax1.set_ylabel('Price ($)', fontsize=11)
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(alpha=0.3)
    
    # ========================================================================
    # 2. Bollinger Bands Width and Price Position
    # ========================================================================
    ax2 = axes[1]
    
    # Plot BB Width percentage
    ax2.plot(ticker_data.index, ticker_data['BB_WIDTH_PCT'], 
            label='BB Width (% of Middle)', 
            linewidth=1.8, color='purple', alpha=0.8)
    
    # Mark squeeze zones (width at historical lows)
    if 'BB_WIDTH_PCT' in ticker_data.columns:
        width_ma = ticker_data['BB_WIDTH_PCT'].rolling(window=125).mean()
        squeeze_threshold = width_ma * 0.5
        squeeze_zones = ticker_data['BB_WIDTH_PCT'] < squeeze_threshold
        
        ax2.fill_between(ticker_data.index, 0, ticker_data['BB_WIDTH_PCT'],
                        where=squeeze_zones, alpha=0.2, color='orange',
                        label='Squeeze Zone')
    
    # Secondary axis for %B (price position in bands)
    ax2_b = ax2.twinx()
    percent_b = ticker_data['PERCENT_B']
    ax2_b.plot(ticker_data.index, percent_b, 
              label='%B (Price Position)', 
              linewidth=1.5, color='blue', alpha=0.6, linestyle='-')
    ax2_b.axhline(y=0, color='green', linestyle='--', alpha=0.5, linewidth=1)
    ax2_b.axhline(y=1, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax2_b.axhline(y=0.5, color='gray', linestyle='-', alpha=0.3, linewidth=1)
    
    ax2.set_title('Bollinger Bands Metrics', fontsize=12, fontweight='bold')
    ax2.set_ylabel('BB Width (%)', fontsize=10, color='purple')
    ax2_b.set_ylabel('%B (0=Lower, 1=Upper)', fontsize=10, color='blue')
    ax2_b.set_ylim(-0.2, 1.2)
    
    # Combine legends
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_b.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=9)
    ax2.grid(alpha=0.3)
    
    # ========================================================================
    # 3. Candlestick Patterns and Body Metrics
    # ========================================================================
    ax3 = axes[2]
    
    # Plot candle body size as percentage of price
    ax3.plot(ticker_data.index, ticker_data['CANDLE_BODY_PCT'], 
            label='Candle Body (% of Price)', 
            linewidth=1.5, color='blue', alpha=0.7)
    
    # Plot body-to-range ratio
    ax3.plot(ticker_data.index, ticker_data['BODY_TO_RANGE'] * 100,
            label='Body/Range Ratio (%)', 
            linewidth=1.5, color='orange', alpha=0.7)
    
    # Highlight Bullish Engulfing patterns
    for date in bull_engulf_dates:
        if date in ticker_data.index:
            ax3.axvline(x=date, color='green', alpha=0.2, 
                       linestyle='-', linewidth=2)
            ax3.scatter(date, ticker_data.loc[date, 'BODY_TO_RANGE'] * 100,
                       color='green', s=100, marker='^', 
                       edgecolors='darkgreen', linewidth=1.5,
                       alpha=0.8, zorder=5)
    
    ax3.axhline(y=50, color='gray', linestyle='--', alpha=0.4, linewidth=1)
    ax3.set_title('Candlestick Pattern Metrics', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Percentage (%)', fontsize=10)
    ax3.legend(loc='best', fontsize=9)
    ax3.grid(alpha=0.3)
    
    # ========================================================================
    # 4. RSI and Volume
    # ========================================================================
    ax4 = axes[3]
    
    # Create secondary y-axis for volume
    ax4_vol = ax4.twinx()
    
    # Plot RSI if available
    if 'RSI' in ticker_data.columns:
        ax4.plot(ticker_data.index, ticker_data['RSI'], 
                label='RSI (14)', linewidth=1.8, color='purple', alpha=0.8)
        
        if use_rsi_filter:
            ax4.axhline(y=rsi_oversold, color='green', linestyle='--', 
                       alpha=0.6, linewidth=1.5,
                       label=f'Entry Threshold ({rsi_oversold})')
        
        ax4.axhline(y=70, color='red', linestyle='--', 
                   alpha=0.4, linewidth=1, label='Overbought (70)')
        ax4.axhline(y=30, color='green', linestyle='--', 
                   alpha=0.4, linewidth=1, label='Oversold (30)')
        ax4.axhline(y=50, color='gray', linestyle='-', 
                   alpha=0.3, linewidth=1)
        ax4.set_ylabel('RSI', fontsize=10)
        ax4.set_ylim(0, 100)
    
    # Plot volume bars
    if 'Volume' in ticker_data.columns:
        colors = ['green' if c >= o else 'red' 
                 for c, o in zip(ticker_data['Close'], ticker_data['Open'])]
        ax4_vol.bar(ticker_data.index, ticker_data['Volume'], 
                   alpha=0.3, color=colors, width=0.8)
        
        # Volume moving average
        vol_ma = ticker_data['Volume'].rolling(window=20).mean()
        ax4_vol.plot(ticker_data.index, vol_ma, 
                    color='blue', linewidth=1.5, 
                    label='Volume MA(20)', alpha=0.7)
        ax4_vol.set_ylabel('Volume', fontsize=10)
    
    ax4.set_title('RSI and Volume Analysis', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Date', fontsize=10)
    
    # Combine legends
    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4_vol.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=9)
    ax4.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    


def _add_visualization_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add additional metrics for visualization.
    
    Args:
        df: DataFrame with OHLC and BB data
        
    Returns:
        DataFrame with additional metrics
    """
    result = df.copy()
    
    # %B (Price position within bands)
    # 0 = at lower band, 1 = at upper band, 0.5 = at middle
    band_range = result['BB_UPPER'] - result['BB_LOWER']
    band_range = band_range.replace(0, np.nan)
    result['PERCENT_B'] = (result['Close'] - result['BB_LOWER']) / band_range
    
    # Candle body metrics
    result['CANDLE_BODY'] = result['Close'] - result['Open']
    result['CANDLE_BODY_ABS'] = abs(result['CANDLE_BODY'])
    result['CANDLE_BODY_PCT'] = (result['CANDLE_BODY_ABS'] / result['Close'] * 100)
    
    # Body to range ratio
    candle_range = result['High'] - result['Low']
    candle_range = candle_range.replace(0, np.nan)
    result['BODY_TO_RANGE'] = result['CANDLE_BODY_ABS'] / candle_range
    
    # Upper and lower shadows
    result['UPPER_SHADOW'] = result['High'] - result[['Open', 'Close']].max(axis=1)
    result['LOWER_SHADOW'] = result[['Open', 'Close']].min(axis=1) - result['Low']
    
    return result

