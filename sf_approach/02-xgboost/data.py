from dataclasses import dataclass
from datetime import datetime

@dataclass
class PredictItem:
    timestamp: datetime
    stock_price: float
    direction: float

@dataclass
class BacktestItem:
    timestamp: datetime
    action: int
    stock_price: float
    portfolio_value: float
    position: int
    cash: float
    cash_short: float
    cash_collateral:float