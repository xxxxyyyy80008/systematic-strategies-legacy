"""
MABW Strategy Signal Visualization.

"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Optional
from matplotlib.gridspec import GridSpec

def plot_mabw_signals(data: pd.DataFrame, 
                     ticker: str,
                     results,
                     start_date: str = None, 
                     end_date: str = None,
                     strategy_config = None):
    """
    Plot MABW strategy signals for a selected ticker.
    
    Args:
        data: DataFrame with OHLCV and MABW indicators (from strategy.prepare_data())
        ticker: Ticker symbol to plot
        results: BacktestResults object from engine.run()
        start_date: Start date (format: 'YYYY-MM-DD')
        end_date: End date (format: 'YYYY-MM-DD')
        strategy_config: StrategyConfig object or dict with strategy parameters
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
    
    # Extract trades for this ticker from results
    all_trades = results.trades
    ticker_trades = all_trades[all_trades['ticker'] == ticker].copy()
    
    # Separate entry and exit trades
    entries = ticker_trades[ticker_trades['action'] == 'OPEN'].copy()
    exits = ticker_trades[ticker_trades['action'] == 'CLOSE'].copy()
    
    # Apply date filter to trades
    if start_date:
        entries = entries[pd.to_datetime(entries['execution_date']) >= start_date]
        exits = exits[pd.to_datetime(exits['execution_date']) >= start_date]
    if end_date:
        entries = entries[pd.to_datetime(entries['execution_date']) <= end_date]
        exits = exits[pd.to_datetime(exits['execution_date']) <= end_date]
    
    # Extract strategy config parameters
    if strategy_config is not None:
        if hasattr(strategy_config, 'parameters'):
            params = strategy_config.parameters
        else:
            params = strategy_config
    else:
        params = {}
    
    ema_period = params.get('ema_period', 20)
    mabw_lookback = params.get('mabw_llv_period', 10)
    atr_multiplier = params.get('atr_multiplier', 3.0)
    mab_width_critical = params.get('mab_width_critical', 30)
    
    # Print header
    print(f"\n" + "="*80)
    print(f"MABW STRATEGY SIGNALS FOR {ticker}")
    if start_date or end_date:
        date_range = f"{start_date if start_date else ticker_data.index[0].strftime('%Y-%m-%d')} to {end_date if end_date else ticker_data.index[-1].strftime('%Y-%m-%d')}"
        print(f"Period: {date_range}")
    print("="*80)
    
    # Create figure with subplots
    fig, axes = plt.subplots(5, 1, figsize=(16, 18), 
                             gridspec_kw={'height_ratios': [3, 2, 2, 2, 1.5]})
    
    # === SUBPLOT 1: Price with MABW Bands and EMA ===
    ax1 = axes[0]
    
    # Plot price and MABW bands
    ax1.plot(ticker_data.index, ticker_data['Close'], 
             label='Close Price', linewidth=1.5, color='black', alpha=0.7, zorder=2)
    
    if 'EMA' in ticker_data.columns:
        ax1.plot(ticker_data.index, ticker_data['EMA'], 
                 label=f'EMA ({ema_period})', 
                 linewidth=1.5, color='orange', alpha=0.8, zorder=2)
    
    ax1.plot(ticker_data.index, ticker_data['MAB_UPPER'], 
             label='MAB Upper', linewidth=1.0, color='red', alpha=0.6, linestyle='--', zorder=1)
    ax1.plot(ticker_data.index, ticker_data['MAB_MIDDLE'], 
             label='MAB Middle', linewidth=1.0, color='blue', alpha=0.6, linestyle='--', zorder=1)
    ax1.plot(ticker_data.index, ticker_data['MAB_LOWER'], 
             label='MAB Lower', linewidth=1.0, color='green', alpha=0.6, linestyle='--', zorder=1)
    
    # Fill between bands
    ax1.fill_between(ticker_data.index, ticker_data['MAB_UPPER'], ticker_data['MAB_LOWER'],
                     alpha=0.1, color='gray', label='MABW Band', zorder=0)
    
    # Plot Entry signals from results.trades
    if len(entries) > 0:
        for idx, trade in entries.iterrows():
            exec_date = pd.to_datetime(trade['execution_date'])
            if exec_date in ticker_data.index:
                price = trade['price']
                # Extract signal type/name if available
                signal_name = trade.get('signal_name', '')
                signal_color = 'darkgreen' if 'RULE2' in str(signal_name).upper() else 'green'
                ax1.scatter(exec_date, price, color=signal_color, s=100, marker='^', 
                           edgecolors='darkgreen', linewidths=1.5, zorder=5)
    
    # Add legend label for entries
    ax1.scatter([], [], color='green', s=100, marker='^', 
               edgecolors='darkgreen', linewidths=1.5, label='Entry')
    
    # Plot Exit signals from results.trades
    if len(exits) > 0:
        for idx, trade in exits.iterrows():
            exec_date = pd.to_datetime(trade['execution_date'])
            if exec_date in ticker_data.index:
                price = trade['price']
                # Extract signal type/name if available
                signal_name = trade.get('signal_name', '')
                signal_color = 'darkred' if 'TRAILING' in str(signal_name).upper() else 'red'
                ax1.scatter(exec_date, price, color=signal_color, s=100, marker='v', 
                           edgecolors='darkred', linewidths=1.5, zorder=5)
    
    # Add legend label for exits
    ax1.scatter([], [], color='red', s=100, marker='v', 
               edgecolors='darkred', linewidths=1.5, label='Exit')
    
    ax1.set_title(f'{ticker} - MABW Strategy Signals', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Price ($)', fontsize=11, fontweight='bold')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(alpha=0.3, linestyle='--')
    
    # === SUBPLOT 2: MABW Width and LLV ===
    ax2 = axes[1]
    
    ax2.plot(ticker_data.index, ticker_data['MAB_WIDTH'], 
            label='MABW Width (%)', linewidth=1.5, color='purple')
    
    if 'MAB_LLV' in ticker_data.columns:
        ax2.plot(ticker_data.index, ticker_data['MAB_LLV'], 
                label=f'MABW LLV ({mabw_lookback}d)', 
                linewidth=1.5, color='orange', alpha=0.7)
    
    # Add critical threshold line
    if mab_width_critical:
        ax2.axhline(y=mab_width_critical, color='red', 
                   linestyle='--', alpha=0.7, linewidth=1.5,
                   label=f'Critical ({mab_width_critical})')
    
    # Mark entry signals on width chart
    if len(entries) > 0:
        for idx, trade in entries.iterrows():
            exec_date = pd.to_datetime(trade['execution_date'])
            if exec_date in ticker_data.index:
                width_val = ticker_data.loc[exec_date, 'MAB_WIDTH']
                ax2.scatter(exec_date, width_val, marker='^', s=80, 
                           c='lime', edgecolors='darkgreen', linewidths=1.5, zorder=5)
    
    ax2.set_title('MABW Width and LLV', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Width (%)', fontsize=10, fontweight='bold')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(alpha=0.3, linestyle='--')
    ax2.set_ylim(bottom=0)
    
    # === SUBPLOT 3: ATR and Trailing Stop Levels ===
    ax3 = axes[2]
    
    if 'ATR' in ticker_data.columns:
        ax3.plot(ticker_data.index, ticker_data['ATR'], 
                label='ATR', linewidth=1.5, color='blue')
        atr_trailing = ticker_data['ATR'] * atr_multiplier
        ax3.plot(ticker_data.index, atr_trailing, 
                label=f'ATR × {atr_multiplier} (Trailing Stop Distance)', 
                linewidth=1.5, color='red', alpha=0.7)
    
    ax3.set_title('Average True Range (ATR)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('ATR ($)', fontsize=10, fontweight='bold')
    ax3.legend(loc='best', fontsize=9)
    ax3.grid(alpha=0.3, linestyle='--')
    
    # === SUBPLOT 4: Price vs EMA vs Bands Analysis ===
    ax4 = axes[3]
    
    price_to_upper = (ticker_data['Close'] - ticker_data['MAB_UPPER']) / ticker_data['MAB_UPPER'] * 100
    price_to_lower = (ticker_data['Close'] - ticker_data['MAB_LOWER']) / ticker_data['MAB_LOWER'] * 100
    
    ax4.plot(ticker_data.index, price_to_upper, 
            label='Price to Upper Band (%)', linewidth=1.5, color='red', alpha=0.7)
    ax4.plot(ticker_data.index, price_to_lower, 
            label='Price to Lower Band (%)', linewidth=1.5, color='green', alpha=0.7)
    
    if 'EMA' in ticker_data.columns:
        ema_to_upper = (ticker_data['EMA'] - ticker_data['MAB_UPPER']) / ticker_data['MAB_UPPER'] * 100
        ax4.plot(ticker_data.index, ema_to_upper, 
                label='EMA to Upper Band (%)', linewidth=1.5, color='orange', alpha=0.7)
    
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
    ax4.set_title('Price and EMA Position Relative to Bands', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Distance (%)', fontsize=10, fontweight='bold')
    ax4.legend(loc='best', fontsize=9)
    ax4.grid(alpha=0.3, linestyle='--')
    
    # === SUBPLOT 5: Signal Visualization ===
    ax5 = axes[4]
    
    # Create signal series from trades
    signal_series = pd.Series(0, index=ticker_data.index)
    
    # Mark entries as 1
    for idx, trade in entries.iterrows():
        exec_date = pd.to_datetime(trade['execution_date'])
        if exec_date in signal_series.index:
            signal_series.loc[exec_date] = 1
    
    # Mark exits as -1
    for idx, trade in exits.iterrows():
        exec_date = pd.to_datetime(trade['execution_date'])
        if exec_date in signal_series.index:
            signal_series.loc[exec_date] = -1
    
    # Plot signals
    colors = signal_series.map({1: 'green', -1: 'red', 0: 'gray'})
    ax5.scatter(ticker_data.index, signal_series, c=colors, s=20, alpha=0.7)
    ax5.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
    ax5.set_yticks([-1, 0, 1])
    ax5.set_yticklabels(['Exit (-1)', 'No Signal (0)', 'Entry (1)'])
    ax5.set_title('Trading Signals', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Date', fontsize=10, fontweight='bold')
    ax5.grid(alpha=0.3, axis='x', linestyle='--')
    ax5.set_ylim([-1.5, 1.5])
    
    plt.tight_layout()
    plt.show()
    
    # === Print Strategy Statistics ===
    print(f"\nMABW Strategy Statistics for {ticker}:")
    print("-" * 80)
    print(f"  Total Entry Signals: {len(entries)}")
    print(f"  Total Exit Signals: {len(exits)}")
    
    # Count signals by type if available
    if 'signal_name' in entries.columns:
        entry_counts = entries['signal_name'].value_counts()
        if len(entry_counts) > 0:
            print("\n  Entry Signal Types:")
            for signal_type, count in entry_counts.items():
                print(f"    {signal_type}: {count}")
    
    if 'signal_name' in exits.columns:
        exit_counts = exits['signal_name'].value_counts()
        if len(exit_counts) > 0:
            print("\n  Exit Signal Types:")
            for signal_type, count in exit_counts.items():
                print(f"    {signal_type}: {count}")
    
    # Trade performance for this ticker
    if len(exits) > 0:
        total_pnl = exits['net_pnl'].sum()
        wins = exits[exits['net_pnl'] > 0]
        losses = exits[exits['net_pnl'] <= 0]
        win_rate = len(wins) / len(exits) * 100 if len(exits) > 0 else 0
        
        print(f"\n  Trade Performance:")
        print(f"    Total P&L: ${total_pnl:,.2f}")
        print(f"    Wins: {len(wins)} | Losses: {len(losses)}")
        print(f"    Win Rate: {win_rate:.1f}%")
        
        if len(wins) > 0:
            print(f"    Avg Win: ${wins['net_pnl'].mean():,.2f}")
        if len(losses) > 0:
            print(f"    Avg Loss: ${losses['net_pnl'].mean():,.2f}")
    
    print("="*80)
