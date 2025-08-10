from data import PredictItem, BacktestItem, PortfolioState, Action
from collections import deque

def _process_trade_item(
    item: PredictItem,
    state: PortfolioState,
    action_threshold: int,
    buy_rate: float,
    short_rate: float
) -> BacktestItem:

    current_price = item.stock_price
    current_date = item.timestamp.date()
    current_direction = item.direction

    opened_today = False

    last_action:Action | None = None
    last_action_amount:int | None = None

    log_list:list[str] = []

    # first we remove the old trade dates
    while state.intraday_trade_dates and (current_date - state.intraday_trade_dates[0]).days >= 5:
        log_list.append(f"removing 1 entry from state.intraday_trade_dates (count {len(state.intraday_trade_dates)}, oldest date {state.intraday_trade_dates[0]}")
        state.intraday_trade_dates.popleft()

    # margin call check prep
    short_market_value = abs(state.position) * current_price if state.position < 0 else 0
    required_margin = 0.3 * short_market_value
    collateral_cash = state.cash - state.cash_short

    # Intraday rule check
    if len(state.intraday_trade_dates) >= 3 and current_date == state.last_trade_date:
        log_list.append(f"intraday stop triggered - trades {state.intraday_trade_dates}")
        portfolio_value = state.cash + state.position * current_price
        

    # Partial margin call check
    elif state.position < 0 and collateral_cash < required_margin:
        required_short_value = collateral_cash / 0.3
        allowed_position = required_short_value / current_price
        shares_to_cover = int(min(abs(state.position) - allowed_position, abs(state.position)))

        log_list.append(f"30% margin call")
        state.cash -= shares_to_cover * current_price
        state.cash_short -= shares_to_cover * current_price
        state.position += shares_to_cover
        last_action = Action.MARGIN_CALL
        last_action_amount = shares_to_cover

    # normal trading
    else: 
        # LONG entry
        if current_direction > action_threshold:

            # close the short if any first
            if state.position < 0:
                log_list.append(f"LONG:close short")
                old_pos = state.position
                state.cash -= abs(state.position) * current_price
                state.cash_short = 0
                state.position = 0
                last_action = Action.CLOSE_SHORT
                last_action_amount = old_pos

            # buy more shares...
            elif state.position == 0:
                to_invest = buy_rate * state.cash
                num_shares = int(to_invest // current_price)

                if num_shares > 0:
                    log_list.append(f"LONG:buys shares")
                    state.cash -= num_shares * current_price
                    state.position += num_shares
                    last_action = Action.OPEN_LONG
                    last_action_amount = num_shares
                    opened_today = True

        # SHORT entry
        elif current_direction < -action_threshold:

            # we sell the shares we own
            if state.position > 0:
                log_list.append(f"SHORT:sell shares")
                last_action = Action.CLOSE_LONG
                last_action_amount = -state.position

                state.cash += state.position * current_price
                state.position = 0
                
            # we open a short position otherwise
            elif state.position == 0:
                collateral_cash = state.cash - state.cash_short
                shortable_amount = collateral_cash // 2
                shortable_shares_max = int(shortable_amount // current_price)

                want_short_amount = short_rate * state.cash
                want_short_shares = int(want_short_amount // current_price)

                num_shorts = min(want_short_shares, shortable_shares_max)
                if num_shorts > 0:
                    log_list.append(f"SHORT:open shorts")
                    state.cash += num_shorts * current_price
                    state.cash_short += num_shorts * current_price
                    state.position -= num_shorts
                    opened_today = True
                    last_action = Action.OPEN_SHORT
                    last_action_amount = -num_shorts

    # Intraday detection
    if last_action is not None and current_date == state.last_trade_date and opened_today:
        state.intraday_trade_dates.append(current_date)

    if last_action is not None:
        state.last_trade_date = current_date

    portfolio_value = state.cash + state.position * current_price

    return BacktestItem(
        timestamp=item.timestamp,
        direction=item.direction,
        log=",".join(log_list),
        trade_action=last_action,
        trade_amount=last_action_amount,
        stock_price=current_price,
        portfolio_value=portfolio_value,
        position=state.position,
        cash=state.cash,
        cash_short=state.cash_short,
        cash_collateral=state.cash - state.cash_short,
    )


def run_backtest(
        predict_item_list:list[PredictItem], 
        action_threshold:int, 
        initial_cash:int = 100000,
        buy_rate = 0.2, 
        short_rate = 0.2) -> list[BacktestItem]:

    backtest_list:list[BacktestItem] = []

    state = PortfolioState(
        cash=initial_cash,
        cash_short=0,
        position=0,
        last_trade_date=None,
        last_action=0,
        intraday_trade_dates=deque()
    )

    for item in predict_item_list:

        result = _process_trade_item(
            item=item,
            state=state,
            action_threshold=action_threshold,
            buy_rate=buy_rate,
            short_rate=short_rate
        )

        backtest_list.append(result)

    return backtest_list