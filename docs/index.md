# Crypto Backtester

A backtesting framework for cryptocurrency trading strategies.

## Quick Start

```python
from datetime import datetime, timedelta, timezone
from crypto_backtester_binance import Backtester, PositionManager
from backtest.example.v1_hold import HoldStrategy

bt = Backtester(historical_data_dir="./historical_data")
strategy = HoldStrategy(symbols=["BTC-USDT","ETH-USDT"], lookback_days=0)
pm = PositionManager()
start_date = datetime.now(timezone.utc) - timedelta(days=30)
end_date = datetime.now(timezone.utc)

results = bt.run_backtest(
    strategy=strategy,
    position_manager=pm,
    start_date=start_date,
    end_date=end_date,
    time_step=timedelta(hours=1),
    market_type="futures",
)
```

## Installation

```bash
pip install -e ".[docs]"
```

## Building Documentation

```bash
mkdocs build
mkdocs serve  # Preview at http://127.0.0.1:8000
```

See [Overview](overview.md) for architecture details.

