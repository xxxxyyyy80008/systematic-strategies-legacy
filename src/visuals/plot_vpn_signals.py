"""
VPN Strategy Visualization
"""
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional


def plot_vpn_signals(data: pd.DataFrame, 
                     ticker: str,
                     strategy_params: dict,
                     start_date: Optional[str] = None, 
                     end_date: Optional[str] = None):
    """
    Plot VPN strategy signals for a selected ticker.
    
    Args:
        combined_data: Combined DataFrame with all tickers and signals
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
    vpn_critical = strategy_params.get('vpn_critical', 5.0)
    vpn_period = strategy_params.get('vpn_period', 30)
    vpn_ma_period = strategy_params.get('vpn_ma_period', 30)
    rsi_max = strategy_params.get('rsi_max_value', 70)
    rsi_min = strategy_params.get('rsi_min_value', 30)
    price_ma_period = strategy_params.get('price_ma_period', 50)
    
    # Print header
    print(f"\n" + "="*80)
    print(f"VPN STRATEGY VISUALIZATION FOR {ticker}")
    if start_date or end_date:
        date_range = f"{start_date or ticker_data.index[0].strftime('%Y-%m-%d')} to {end_date or ticker_data.index[-1].strftime('%Y-%m-%d')}"
        print(f"Period: {date_range}")
    print("="*80)
    
    # Check required columns
    required_cols = ['Close', 'VPN_SMOOTHED', 'VPN_MA', 'RSI', 'PRICE_MA']
    missing_cols = [col for col in required_cols if col not in ticker_data.columns]
    if missing_cols:
        print(f"  Warning: Missing columns: {missing_cols}")
        print("   Make sure to run prepare_data() first")
        return
    
    # Create figure with subplots
    fig, axes = plt.subplots(4, 1, figsize=(16, 14), 
                             gridspec_kw={'height_ratios': [3, 2, 2, 2]})
    
    # ========================================================================
    # 1. Price Chart with Moving Average
    # ========================================================================
    ax1 = axes[0]
    
    # Plot price
    ax1.plot(ticker_data.index, ticker_data['Close'], 
             label='Close Price', linewidth=1.5, color='black', alpha=0.8)
    
    # Plot price moving average
    if 'PRICE_MA' in ticker_data.columns:
        ax1.plot(ticker_data.index, ticker_data['PRICE_MA'], 
                label=f'Price MA({price_ma_period})', 
                linewidth=1.2, color='blue', alpha=0.6, linestyle='--')
    
    # Mark entry conditions (Price > MA, VPN > Critical, RSI < Max)
    entry_conditions = (
        (ticker_data['Close'] > ticker_data['PRICE_MA']) &
        (ticker_data['VPN_SMOOTHED'] >= vpn_critical) &
        (ticker_data['RSI'] < rsi_max)
    )
    entry_dates = ticker_data[entry_conditions].index
    
    if len(entry_dates) > 0:
        ax1.scatter(entry_dates, ticker_data.loc[entry_dates, 'Close'],
                   color='green', s=100, marker='^', 
                   edgecolors='darkgreen', linewidth=1.5,
                   label='Entry Condition Met', zorder=5, alpha=0.7)
    
    # Mark exit conditions (VPN crosses below MA)
    if 'VPN_MA' in ticker_data.columns:
        exit_conditions = (
            (ticker_data['VPN_SMOOTHED'] < ticker_data['VPN_MA']) &
            (ticker_data['VPN_SMOOTHED'].shift(1) >= ticker_data['VPN_MA'].shift(1)) &
            (ticker_data['RSI'] > rsi_min)
        )
        exit_dates = ticker_data[exit_conditions].index
        
        if len(exit_dates) > 0:
            ax1.scatter(exit_dates, ticker_data.loc[exit_dates, 'Close'],
                       color='red', s=100, marker='v', 
                       edgecolors='darkred', linewidth=1.5,
                       label='Exit Condition Met', zorder=5, alpha=0.7)
    
    ax1.set_title(f'{ticker} - VPN Strategy Price Chart', 
                 fontsize=14, fontweight='bold')
    ax1.set_ylabel('Price ($)', fontsize=11)
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(alpha=0.3)
    
    # ========================================================================
    # 2. VPN Indicator with Critical Level
    # ========================================================================
    ax2 = axes[1]
    
    # Plot smoothed VPN
    ax2.plot(ticker_data.index, ticker_data['VPN_SMOOTHED'], 
            label=f'VPN Smoothed({vpn_period})', 
            linewidth=2.0, color='blue')
    
    # Plot VPN moving average
    if 'VPN_MA' in ticker_data.columns:
        ax2.plot(ticker_data.index, ticker_data['VPN_MA'], 
                label=f'VPN MA({vpn_ma_period})', 
                linewidth=1.5, color='orange', alpha=0.7)
    
    # Plot critical level
    ax2.axhline(y=vpn_critical, color='red', linestyle='--', 
               alpha=0.7, linewidth=2,
               label=f'Critical Level ({vpn_critical})')
    
    # Fill above/below critical level
    ax2.fill_between(ticker_data.index, vpn_critical, ticker_data['VPN_SMOOTHED'], 
                     where=(ticker_data['VPN_SMOOTHED'] >= vpn_critical), 
                     alpha=0.2, color='green', label='Bullish Zone')
    ax2.fill_between(ticker_data.index, vpn_critical, ticker_data['VPN_SMOOTHED'], 
                     where=(ticker_data['VPN_SMOOTHED'] < vpn_critical), 
                     alpha=0.2, color='red', label='Bearish Zone')
    
    # Mark crossovers
    vpn_cross_above = (
        (ticker_data['VPN_SMOOTHED'] >= vpn_critical) &
        (ticker_data['VPN_SMOOTHED'].shift(1) < vpn_critical)
    )
    cross_above_dates = ticker_data[vpn_cross_above].index
    if len(cross_above_dates) > 0:
        ax2.scatter(cross_above_dates, 
                   ticker_data.loc[cross_above_dates, 'VPN_SMOOTHED'],
                   color='green', s=80, marker='^', 
                   edgecolors='darkgreen', linewidth=1,
                   label='Cross Above Critical', zorder=5)
    
    ax2.set_title('Volume Pressure Number (VPN) Indicator', 
                 fontsize=12, fontweight='bold')
    ax2.set_ylabel('VPN', fontsize=10)
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(alpha=0.3)
    
    # ========================================================================
    # 3. Volume Analysis
    # ========================================================================
    ax3 = axes[2]
    
    if 'Volume' in ticker_data.columns:
        # Create secondary y-axis for volume bars
        ax3_vol = ax3.twinx()
        
        # Plot volume bars
        if 'Open' in ticker_data.columns:
            colors = ['green' if close >= open_price else 'red' 
                     for close, open_price in zip(ticker_data['Close'], 
                                                   ticker_data['Open'])]
        else:
            colors = ['gray'] * len(ticker_data)
        
        ax3_vol.bar(ticker_data.index, ticker_data['Volume'], 
                   alpha=0.3, color=colors, label='Volume', width=0.8)
        
        # Add volume moving average if available
        if 'Volume' in ticker_data.columns:
            vol_ma = ticker_data['Volume'].rolling(window=20).mean()
            ax3.plot(ticker_data.index, vol_ma, 
                    label='Volume MA(20)', 
                    linewidth=1.5, color='blue', alpha=0.7)
        
        ax3.set_title('Volume Analysis', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Volume MA', fontsize=10)
        ax3_vol.set_ylabel('Daily Volume', fontsize=10)
        
        # Combine legends
        lines1, labels1 = ax3.get_legend_handles_labels()
        lines2, labels2 = ax3_vol.get_legend_handles_labels()
        ax3.legend(lines1 + lines2, labels1 + labels2, 
                  loc='best', fontsize=9)
    else:
        ax3.text(0.5, 0.5, 'Volume data not available', 
                ha='center', va='center', transform=ax3.transAxes)
    
    ax3.grid(alpha=0.3)
    
    # ========================================================================
    # 4. RSI Indicator
    # ========================================================================
    ax4 = axes[3]
    
    # Plot RSI
    ax4.plot(ticker_data.index, ticker_data['RSI'], 
            label='RSI(14)', linewidth=1.5, color='purple')
    
    # Reference lines
    ax4.axhline(y=rsi_max, color='red', linestyle='--', 
               alpha=0.6, linewidth=1.5,
               label=f'Max Entry Level ({rsi_max})')
    ax4.axhline(y=50, color='gray', linestyle='-', 
               alpha=0.3, linewidth=1)
    ax4.axhline(y=70, color='orange', linestyle='--', 
               alpha=0.4, linewidth=1, label='Overbought (70)')
    ax4.axhline(y=30, color='green', linestyle='--', 
               alpha=0.4, linewidth=1, label='Oversold (30)')
    
    # Fill overbought/oversold zones
    ax4.fill_between(ticker_data.index, 70, 100, 
                     alpha=0.1, color='red', label='')
    ax4.fill_between(ticker_data.index, 0, 30, 
                     alpha=0.1, color='green', label='')
    
    ax4.set_title('Relative Strength Index (RSI)', 
                 fontsize=12, fontweight='bold')
    ax4.set_ylabel('RSI', fontsize=10)
    ax4.set_xlabel('Date', fontsize=10)
    ax4.set_ylim(0, 100)
    ax4.legend(loc='best', fontsize=9)
    ax4.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
