from data import PredictItem, BacktestItem

def run_backtest(
        predict_item_list:list[PredictItem], 
        action_threshold:int, 
        initial_cash:int = 100000, 
        enable_intraday:bool = False, 
        buy_rate = 0.2, 
        short_rate = 0.2) -> list[BacktestItem]:

    # Portfolio and tracking
    cash = initial_cash
    cash_short = 0
    position = 0
    last_trade_date = False
    last_action = 0

    ret_list:list[BacktestItem] = []

    for item in predict_item_list:

        current_price = item.stock_price
        current_date = item.timestamp.date()
        current_direction = item.direction
        
        # Partial Margin Call Check
        if position < 0:
            short_market_value = abs(position) * current_price
            required_margin = 0.3 * short_market_value
            collateral_cash = cash - cash_short

            if collateral_cash < required_margin:
                # Calculate how many shares to cover to fix margin
                required_short_value = collateral_cash / 0.3
                allowed_position = required_short_value / current_price
                shares_to_cover = abs(position) - allowed_position
                shares_to_cover = int(min(shares_to_cover, abs(position)))  # clip to max available

                if shares_to_cover > 0:
                    #print(f"** MARGIN CALL: partially covering {shares_to_cover} shares at {current_price} = {shares_to_cover * current_price}")
                    cash -= shares_to_cover * current_price
                    cash_short -= shares_to_cover * current_price
                    position += shares_to_cover  # reduce negative short position
                    last_trade_date = current_date
                    continue


        # Skip if a trade was already made today (simulate T+1 rule)
        if not enable_intraday and current_date == last_trade_date:
            #print(f"Skipping trade because last_trade_date = {last_trade_date}")
            portfolio_value = cash + position * current_price
            continue

        # strong LONG entry
        if current_direction > action_threshold:

            # exit ALL short first
            if position < 0:
                cash -= abs(position) * current_price
                cash_short = 0
                position = 0
                last_trade_date = current_date
            else:
                # buy shares more aggressively if we can
                to_invest = buy_rate * cash
                num_shares = to_invest // current_price
                if num_shares > 0:

                    #print(f"** BOUGHT {num_shares} at {current_price} = {num_shares * current_price}")
                    cash -= num_shares * current_price
                    position += num_shares
                    entry_price = current_price
                    last_trade_date = current_date

        # strong short
        elif current_direction < -action_threshold:
            
            # we sell 100% of our shares
            if position > 0:

                to_sell = position

                #print(f"** SOLD {to_sell} at {current_price} = {to_sell * current_price}")
                cash += to_sell * current_price
                position -= to_sell
                last_trade_date = current_date

            else:
                # then we go short (SHORT_RATE% of cash)
                collateral_cash = cash - cash_short

                shortable_amount = collateral_cash// 2 #50% margin
                shortable_shares_max = shortable_amount // current_price

                want_short_amount = short_rate * cash
                want_short_shares = want_short_amount // current_price

                num_shorts = min(want_short_shares, shortable_shares_max)
                if num_shorts > 0:

                    #print(f"** SHORTING {num_shorts} at {current_price} = {num_shorts * current_price}")
                    cash += num_shorts * current_price
                    cash_short += num_shorts * current_price

                    position -= num_shorts  # Negative value = short position
                    last_trade_date = current_date

        # Track portfolio value
        portfolio_value = cash + position * current_price

        #print(f"-> current position: {position} / portfolio_value {portfolio_value}")

        ret_list.append(BacktestItem(
            timestamp=item.timestamp,
            action=last_action,
            stock_price=current_price,
            portfolio_value=portfolio_value,
            position=position,
            cash=cash,
            cash_short=cash_short,
        ))

    return ret_list