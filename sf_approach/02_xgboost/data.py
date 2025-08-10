from dataclasses import dataclass
from datetime import datetime
from datetime import date
from collections import deque

from enum import Enum

class Action(str, Enum):
    CLOSE_SHORT = "close_short"
    CLOSE_LONG = "close_long"
    OPEN_LONG = "open_long"
    OPEN_SHORT = "open_short"
    MARGIN_CALL = "margin_call"

@dataclass
class PredictItem:
    timestamp: datetime
    stock_price: float
    direction: float

@dataclass
class BacktestItem:
    timestamp: datetime
    direction: float
    stock_price: float
    portfolio_value: float
    position: int
    cash: float
    cash_short: float
    cash_collateral:float
    log: str
    trade_action: Action | None = None
    trade_amount: int | None = None


@dataclass
class PortfolioState:
    cash: float
    cash_short: float
    position: int
    last_action: int
    intraday_trade_dates: deque
    last_trade_date: date | None