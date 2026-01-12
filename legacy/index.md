---
layout: default
title: Strategies & Backtests
nav_order: 1
has_children: true
has_toc: false
permalink: /systematic-strategies-legacy/legacy/
---

# Systematic Trading Strategies & Backtesting
{: .fs-7 }

Systematic rule-based strategies, backtesting, and performance evaluation.
{: .fs-5 .fw-300 }

## Strategy Catalog

### Mean Reversion

| Strategy | Description | Link |
|:---------|:------------|:-----|
| **Bollinger Mean-Reversion** | Classic mean-reversion with z-score entry/exit | [View](/systematic-strategies-legacy/legacy/c_001_tasc201905/c_001_tasc201905.html) |
| **Bollinger + Candlesticks** | Band signals with pattern confirmation | [View](/dsystematic-strategies-legacy/legacy/c_002_tasc201910/c_002_tasc201910.html) |

---

### Trend Following

| Strategy | Description | Link |
|:---------|:------------|:-----|
| **Adaptive Moving Averages** | KAMA/FRAMA-based trend detection | [View](/dsystematic-strategies-legacy/legacy/c_003_tasc201804/c_003_tasc201804.html) |
| **Weekly & Daily MACD** | Multi-timeframe trend alignment | [View](/dsystematic-strategies-legacy/legacy/c_007_tasc201712/c_007_tasc201712.html) |
| **MAB/MABW System** | Band width expansion signals | [View](/dsystematic-strategies-legacy/legacy/c_005_tasc202108/c_005_tasc202108.html) |

---

### Breakout Detection

| Strategy | Description | Link |
|:---------|:------------|:-----|
| **VPN High-Volume Breakouts** | Price breakout + volume confirmation | [View](/dsystematic-strategies-legacy/legacy/c_004_tasc202104/c_004_tasc202104.html) |
| **DMI Continuation Signals** | ADX trend strength + directional crossover | [View](/dsystematic-strategies-legacy/legacy/c_011_tasc202212/c_011_tasc202212.html) |

---

### Multi-Factor Composite

| Strategy | Description | Link |
|:---------|:------------|:-----|
| **Soldiers & Crows** | Candlestick patterns + trend filter | [View](/dsystematic-strategies-legacy/legacy/c_006_tasc201710/c_006_tasc201710.html) |
| **Stoch MACD + RS Composite** | Triple confirmation: momentum + relative strength | [View](/dsystematic-strategies-legacy/legacy/d_001_b38-b39-b40/d_001_b38-b39-b40.html) |

---

## Backtesting Framework

-  **Transaction costs** — Commission + spread modeling
-  **Slippage estimation** — Volume-dependent impact
-  **Walk-forward validation** — Rolling out-of-sample testing
-  **Risk metrics** — Sharpe, Sortino, max drawdown, win rate

---

## References

- **TASC Papers** (2010-2025) — Primary strategy sources
- **Kaufman** — Adaptive methods

---

## Related Sections

- [Technical Indicators](/docs/alpha-research/indicators/) — 55+ indicators powering these strategies
- [Regime Analysis](/docs/regime-analysis/) — Market regime detection for strategy selection
```
