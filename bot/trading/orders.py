from dataclasses import dataclass
from typing import List
from forest import (
    DISCORD_MODE, DISCORD_USER_ID, STATIC_TICKERS, MAX_TICKERS, USE_FULL_SHARES,
    DISCORD_TOKEN, USE_SHORT, USE_TICKER_SELECTION, N_BARS, BAR_TIMEFRAME, REWRITE,
    DATA_DIR, api
)
from alpaca.trading.enums import OrderSide, TimeInForce
from forest import TRADE_LOG_FILENAME, TRADE_LOGIC
import market_timer.timer as market_time
import bot.selection.ranking as ranking
import bot.selection.loader as loader
import bot.stuffs.candles as candles
from datetime import datetime
import pandas as pd
import threading
import discord
import logging
import forest
import os

logger = logging.getLogger(__name__)
WAIT_TICKERS = []
WAITLIST_CSV = os.path.join(DATA_DIR, "waitlist.csv")

# =========================
# Waitlist CSV Management
# =========================

def initialize_waitlist_csv():
    """Create waitlist CSV if it doesn't exist"""
    if not os.path.exists(WAITLIST_CSV):
        df = pd.DataFrame(columns=["ticker", "direction", "qty", "price", "predicted_price", "time"])
        df.to_csv(WAITLIST_CSV, index=False)

def add_to_waitlist(ticker: str, direction: str, qty: float, price: float, predicted_price: float):
    """
    Add or update a ticker in the waitlist CSV.
    - If ticker exists with same direction: reset time to 0
    - If ticker exists with different direction: update direction and reset time to 0
    - If ticker doesn't exist: add new entry with time = 0
    """
    initialize_waitlist_csv()
    
    try:
        df = pd.read_csv(WAITLIST_CSV)
    except Exception:
        df = pd.DataFrame(columns=["ticker", "direction", "qty", "price", "predicted_price", "time"])
    
    # Check if ticker already exists
    existing_idx = df[df['ticker'] == ticker].index
    
    if len(existing_idx) > 0:
        # Update existing entry
        idx = existing_idx[0]
        df.loc[idx, 'direction'] = direction
        df.loc[idx, 'qty'] = qty
        df.loc[idx, 'price'] = price
        df.loc[idx, 'predicted_price'] = predicted_price
        df.loc[idx, 'time'] = 0
        logging.info(f"[{ticker}] Updated in waitlist CSV with direction {direction}, time reset to 0")
    else:
        # Add new entry
        new_row = pd.DataFrame([{
            "ticker": ticker,
            "direction": direction,
            "qty": qty,
            "price": price,
            "predicted_price": predicted_price,
            "time": 0
        }])
        df = pd.concat([df, new_row], ignore_index=True)
        logging.info(f"[{ticker}] Added to waitlist CSV with direction {direction}")
    
    df.to_csv(WAITLIST_CSV, index=False)

def remove_from_waitlist(ticker: str):
    """Remove a ticker from the waitlist CSV"""
    if not os.path.exists(WAITLIST_CSV):
        return
    
    try:
        df = pd.read_csv(WAITLIST_CSV)
        df = df[df['ticker'] != ticker]
        df.to_csv(WAITLIST_CSV, index=False)
        logging.info(f"[{ticker}] Removed from waitlist CSV")
    except Exception as e:
        logging.error(f"Error removing {ticker} from waitlist: {e}")

def increment_waitlist_time(ticker: str):
    """Increment the time counter for a ticker in the waitlist"""
    if not os.path.exists(WAITLIST_CSV):
        return
    
    try:
        df = pd.read_csv(WAITLIST_CSV)
        idx = df[df['ticker'] == ticker].index
        if len(idx) > 0:
            df.loc[idx[0], 'time'] = df.loc[idx[0], 'time'] + 1
            df.to_csv(WAITLIST_CSV, index=False)
    except Exception as e:
        logging.error(f"Error incrementing time for {ticker}: {e}")

def should_add_to_waitlist(ticker: str, action: str) -> bool:
    """
    Check if a ticker should be added to the waitlist based on current positions.
    Returns True if should add, False if should skip.
    """
    try:
        pos = api.get_position(ticker)
        pos_qty = float(pos.qty)
    except Exception:
        pos_qty = 0.0
    
    is_owned = pos_qty > 0
    is_shorted = pos_qty < 0
    
    # Don't add BUY if already owned
    if action == "BUY" and is_owned:
        logging.info(f"[{ticker}] Already owned, skipping BUY waitlist entry")
        return False
    
    # Don't add SELL if not owned
    if action == "SELL" and not is_owned:
        logging.info(f"[{ticker}] Not owned, skipping SELL waitlist entry")
        return False
    
    # Don't add SHORT if already shorted
    if action == "SHORT" and is_shorted:
        logging.info(f"[{ticker}] Already shorted, skipping SHORT waitlist entry")
        return False
    
    # Don't add COVER if not shorted
    if action == "COVER" and not is_shorted:
        logging.info(f"[{ticker}] Not shorted, skipping COVER waitlist entry")
        return False
    
    return True

# =========================
# Batch trading (defer orders)
# =========================

@dataclass
class PendingBuy:
    ticker: str
    direction: str
    qty: float
    price: float
    predicted_price: float
    time: int

@dataclass
class PendingSell:
    ticker: str
    direction: str
    qty: float
    price: float
    predicted_price: float
    time: int

@dataclass
class PendingShort:
    ticker: str
    direction: str
    qty: float
    price: float
    predicted_price: float
    time: int

@dataclass
class PendingCover:
    ticker: str
    direction: str
    qty: float
    price: float
    predicted_price: float
    time: int

class TradeBatch:
    """
    When active, buy/sell/short/cover will be queued instead of executed immediately.
    This stabilizes sizing (split cash / slots) across a full ticker run.
    """
    def __init__(self):
        self.active: bool = False
        self.pending_buys: List[PendingBuy] = []
        self.pending_sells: List[PendingSell] = []
        self.pending_shorts: List[PendingShort] = []
        self.pending_covers: List[PendingCover] = []

    def begin(self):
        self.active = True
        self.pending_buys.clear()
        self.pending_sells.clear()
        self.pending_shorts.clear()
        self.pending_covers.clear()

    def end(self):
        self.active = False

TRADE_BATCH = TradeBatch()

def begin_trade_batch():
    TRADE_BATCH.begin()

def end_trade_batch_and_flush():
    """
    Ends batching and executes queued orders.
    Default order: COVER -> SELL -> SHORT -> BUY
    (Conservative: close shorts first, free/realize cash, then open new risk.)
    """
    TRADE_BATCH.end()
    flush_trade_batch()

def check_for_wait(ticker: str, direction: int) -> bool:
    """
    Check if we should execute or wait for a ticker.
    Returns True if should EXECUTE, False if should WAIT.
    """
    try:
        df = candles.fetch_candles_plus_features(
            ticker,
            bars=N_BARS,
            timeframe=BAR_TIMEFRAME,
            rewrite_mode=REWRITE
        )
        decision = market_time.get_execution_decision(df, ticker, 3, "cat", direction)
        if decision == "EXECUTE":
            return True
        else:
            return False
    except Exception as e:
        logging.error(f"[{ticker}] Error in check_for_wait: {e}")
        return True  # Default to execute on error

def flush_trade_batch():
    """
    Process the waitlist CSV or pending lists based on USE_TICKER_SELECTION.
    If USE_TICKER_SELECTION == 2: Process waitlist CSV with market timer
    Otherwise: Process pending lists directly
    """
    if USE_TICKER_SELECTION == 2:
        # Process waitlist CSV
        if not os.path.exists(WAITLIST_CSV):
            logging.info("Waitlist CSV does not exist, nothing to flush")
            return
        
        try:
            df = pd.read_csv(WAITLIST_CSV)
        except Exception as e:
            logging.error(f"Error reading waitlist CSV: {e}")
            return
        
        if df.empty:
            logging.info("Waitlist CSV is empty, nothing to flush")
            return
        
        tickers_to_remove = []
        
        for idx, row in df.iterrows():
            ticker = row['ticker']
            direction_str = row['direction']
            qty = float(row['qty'])
            price = float(row['price'])
            predicted_price = float(row['predicted_price'])
            time = int(row['time'])
            
            # Map direction string to int for market timer
            # BUY and COVER use direction=1, SELL and SHORT use direction=0
            if direction_str in ["BUY", "COVER"]:
                direction_int = 1
            else:  # SELL or SHORT
                direction_int = 0
            
            logging.info(f"[{ticker}] Processing waitlist entry: {direction_str}, time={time}")
            
            # Check if time has reached 7 (timeout)
            if time >= 7:
                logging.info(f"[{ticker}] Timeout reached (time={time}), removing from waitlist")
                tickers_to_remove.append(ticker)
                continue
            
            # Check market timer decision
            should_execute = check_for_wait(ticker, direction_int)
            
            if should_execute:
                logging.info(f"[{ticker}] Market timer says EXECUTE, processing {direction_str}")
                
                # Execute the appropriate action
                try:
                    if direction_str == "BUY":
                        buy_shares(ticker, qty, price, predicted_price)
                    elif direction_str == "SELL":
                        sell_shares(ticker, qty, price, predicted_price)
                    elif direction_str == "SHORT":
                        short_shares(ticker, qty, price, predicted_price)
                    elif direction_str == "COVER":
                        close_short(ticker, qty, price, predicted_price)
                    
                    # Remove from waitlist after successful execution
                    tickers_to_remove.append(ticker)
                except Exception as e:
                    logging.error(f"[{ticker}] Failed to execute {direction_str}: {e}")
                    # Increment time on failure
                    increment_waitlist_time(ticker)
            else:
                logging.info(f"[{ticker}] Market timer says WAIT, incrementing time")
                increment_waitlist_time(ticker)
        
        # Remove executed or timed-out tickers
        for ticker in tickers_to_remove:
            remove_from_waitlist(ticker)
    
    else:
        # Original behavior: process pending lists directly
        # Covers first
        for c in list(TRADE_BATCH.pending_covers):
            try:
                close_short(c.ticker, c.qty, c.price, c.predicted_price)
            except Exception as e:
                logging.error("[%s] Batch COVER failed: %s", c.ticker, e)

        # Then sells
        for s in list(TRADE_BATCH.pending_sells):
            try:
                sell_shares(s.ticker, s.qty, s.price, s.predicted_price)
            except Exception as e:
                logging.error("[%s] Batch SELL failed: %s", s.ticker, e)

        # Then shorts
        for sh in list(TRADE_BATCH.pending_shorts):
            try:
                short_shares(sh.ticker, sh.qty, sh.price, sh.predicted_price)
            except Exception as e:
                logging.error("[%s] Batch SHORT failed: %s", sh.ticker, e)

        # Then buys
        for b in list(TRADE_BATCH.pending_buys):
            try:
                buy_shares(b.ticker, b.qty, b.price, b.predicted_price)
            except Exception as e:
                logging.error("[%s] Batch BUY failed: %s", b.ticker, e)

        TRADE_BATCH.pending_covers.clear()
        TRADE_BATCH.pending_sells.clear()
        TRADE_BATCH.pending_shorts.clear()
        TRADE_BATCH.pending_buys.clear()

# =========================
# Discord bot integration
# =========================

if DISCORD_MODE == "on":

    discord_client = discord.Client()

    @discord_client.event
    async def on_ready():
        logging.info(f"Discord bot logged in as {discord_client.user}")

    def run_discord_bot():
        try:
            discord_client.run(DISCORD_TOKEN)
        except Exception as e:
            logging.error(f"Discord bot failed to run: {e}")

    discord_thread = threading.Thread(target=run_discord_bot, daemon=True)
    discord_thread.start()

def send_discord_order_message(action, ticker, price, predicted_price, extra_info=""):
    message = (f"Order Action: {action}\nTicker: {ticker}\n"
               f"Price: {price:.2f}\nPredicted: {predicted_price:.2f}\n{extra_info}")
    if DISCORD_MODE == "on" and DISCORD_USER_ID:
        async def discord_send_dm():
            try:
                user = await discord_client.fetch_user(int(DISCORD_USER_ID))
                await user.send(message)
                logging.info(f"Sent Discord DM for {ticker} order: {action}")
            except Exception as e:
                logging.error(f"Discord DM failed: {e}")
        discord_client.loop.create_task(discord_send_dm())
    else:
        logging.info("Discord mode is off or DISCORD_USER_ID not set.")

# =========================
# Trading actions
# =========================

def buy_shares(ticker, qty, buy_price, predicted_price):
    logging.info("BUY: %s", ticker)

    if ticker.upper() in STATIC_TICKERS:
        logging.info("Ticker %s is in STATIC_TICKERS. Skipping...", ticker)
        return

    if qty <= 0:
        return

    # ✅ Batch mode with USE_TICKER_SELECTION == 2: add to waitlist CSV
    if TRADE_BATCH.active and USE_TICKER_SELECTION == 2:
        if should_add_to_waitlist(ticker, "BUY"):
            add_to_waitlist(ticker, "BUY", qty, buy_price, predicted_price)
            logging.info("[%s] Added to waitlist CSV for BUY", ticker)
        return
    
    # ✅ Batch mode (other modes): enqueue instead of executing
    if TRADE_BATCH.active:
        TRADE_BATCH.pending_buys.append(PendingBuy(ticker, "BUY", qty, buy_price, predicted_price, 0))
        logging.info("[%s] Queued BUY (batch mode).", ticker)
        return

    try:
        if ranking.cash_below_minimum():
            logging.info("Available cash below $10. BUY rejected.")
            return

        if ranking.max_tickers_reached() and ticker not in ranking.get_owned_tickers():
            logging.info(f"max_tickers limit {MAX_TICKERS} reached. BUY of {ticker} denied.")
            return

        # --- Position state ---
        pos = None
        try:
            pos = api.get_position(ticker)
            pos_qty = float(pos.qty)
        except Exception:
            pos = None
            pos_qty = 0.0

        already_long = pos_qty > 0
        already_short = pos_qty < 0
        abs_long_qty = pos_qty if already_long else 0.0
        abs_short_qty = abs(pos_qty) if already_short else 0.0

        # 1) If already long, do nothing
        if already_long:
            log_qty = int(abs_long_qty) if USE_FULL_SHARES else round(abs_long_qty, 4)
            logging.info(
                "[%s] Already long %s shares. Skipping new BUY to prevent duplicate long position.",
                ticker, log_qty
            )
            return

        # 3) If there is a short, CLOSE it only (no new BUY in this call)
        if already_short and abs_short_qty > 0:
            log_qty = int(abs_short_qty) if USE_FULL_SHARES else round(abs_short_qty, 4)
            logging.info(
                "[%s] Detected short position of %s shares. "
                "Calling close_short to fully COVER; no new BUY in this call.",
                ticker, log_qty
            )
            close_short(ticker, abs_short_qty, buy_price, predicted_price)
            return

        # 2) Flat: no long/short → proceed with normal BUY (position sizing unchanged)
        if DISCORD_MODE == "on":
            send_discord_order_message(
                "BUY", ticker, buy_price, predicted_price,
                extra_info="Buying shares via Discord bot."
            )
        else:
            account = api.get_account()
            available_cash = float(account.cash)
            logging.info("Available cash: %s", available_cash)
            logging.info("Length of tickers: %s", len(forest.TICKERS))
            total_ticker_slots = (len(forest.TICKERS) if forest.TICKERS else 0) + forest.AI_TICKER_COUNT
            logging.info("Total Ticker slots 1: %s", total_ticker_slots)
            total_ticker_slots = max(total_ticker_slots, 1)
            logging.info("Total Ticker slots 2: %s", total_ticker_slots)
            split_cash = available_cash / total_ticker_slots
            logging.info("Split cash: %s", split_cash)

            is_crypto = "/" in ticker
            max_shares = (split_cash // buy_price) if USE_FULL_SHARES else (split_cash / buy_price)
            logging.info("Max shares: %s", max_shares)
            final_qty = min(qty, max_shares)
            logging.info("Final quantity: %s", final_qty)

            if final_qty <= 0:
                logging.info("[%s] Not enough split cash to buy any shares. Skipping.", ticker)
                return

            if not is_crypto:
                if USE_FULL_SHARES:
                    api.submit_order(
                        symbol=ticker,
                        qty=int(final_qty),
                        side='buy',
                        type='market',
                        time_in_force='gtc'
                    )
                else:
                    api.submit_order(
                        symbol=ticker,
                        qty=round(final_qty, 4),
                        side='buy',
                        type='market',
                        time_in_force='day'
                    )
            else:
                api.submit_order(
                    symbol=ticker,
                    notion=split_cash,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.GTC
                )

            log_qty = int(final_qty) if USE_FULL_SHARES else round(final_qty, 4)
            logging.info(
                "[%s] BUY %s at %.2f (Predicted: %.2f)",
                ticker, log_qty, buy_price, predicted_price
            )
            log_trade("BUY", ticker, final_qty, buy_price, predicted_price, None)
            if ticker not in forest.TICKERS and ticker not in forest.AI_TICKERS:
                forest.AI_TICKERS.append(ticker)

    except Exception as e:
        logging.error("[%s] Buy order failed: %s", ticker, e)


def sell_shares(ticker, qty, sell_price, predicted_price):
    logging.info("SELL: %s", ticker)

    if ticker.upper() in STATIC_TICKERS:
        logging.info("Ticker %s is in STATIC_TICKERS. Skipping...", ticker)
        return

    if qty <= 0:
        return

    # ✅ Batch mode with USE_TICKER_SELECTION == 2: add to waitlist CSV
    if TRADE_BATCH.active and USE_TICKER_SELECTION == 2:
        if should_add_to_waitlist(ticker, "SELL"):
            add_to_waitlist(ticker, "SELL", qty, sell_price, predicted_price)
            logging.info("[%s] Added to waitlist CSV for SELL", ticker)
        return
    
    # ✅ Batch mode (other modes): enqueue instead of executing
    if TRADE_BATCH.active:
        TRADE_BATCH.pending_sells.append(PendingSell(ticker, "SELL", qty, sell_price, predicted_price, 0))
        logging.info("[%s] Queued SELL (batch mode).", ticker)
        return

    use_shorts_enabled = False
    try:
        use_shorts_enabled = USE_SHORT
    except NameError:
        use_shorts_enabled = False

    try:
        pos = None
        try:
            pos = api.get_position(ticker)
            pos_qty = float(pos.qty)
        except Exception:
            pos = None
            pos_qty = 0.0

        already_long = pos_qty > 0
        already_short = pos_qty < 0
        abs_long_qty = pos_qty if already_long else 0.0
        abs_short_qty = abs(pos_qty) if already_short else 0.0

        # 1) If long (NOT shorted): sell as usual
        if already_long and abs_long_qty > 0:
            sellable_qty = min(qty, abs_long_qty)
            if sellable_qty <= 0:
                logging.info(f"[{ticker}] No shares to SELL.")
                return

            if DISCORD_MODE == "on":
                send_discord_order_message(
                    "SELL", ticker, sell_price, predicted_price,
                    extra_info="Selling shares via Discord bot."
                )
            else:
                avg_entry = float(pos.avg_entry_price) if pos else 0.0
                if USE_FULL_SHARES:
                    api.submit_order(
                        symbol=ticker,
                        qty=int(sellable_qty),
                        side='sell',
                        type='market',
                        time_in_force='gtc'
                    )
                else:
                    api.submit_order(
                        symbol=ticker,
                        qty=round(sellable_qty, 4),
                        side='sell',
                        type='market',
                        time_in_force='day'
                    )

                pl = (sell_price - avg_entry) * sellable_qty
                log_qty = int(sellable_qty) if USE_FULL_SHARES else round(sellable_qty, 4)
                logging.info(
                    f"[{ticker}] SELL {log_qty} at {sell_price:.2f} "
                    f"(Predicted: {predicted_price:.2f}, P/L: {pl:.2f})"
                )
                log_trade("SELL", ticker, sellable_qty, sell_price, predicted_price, pl)
                try:
                    new_pos = api.get_position(ticker)
                    if float(new_pos.qty) == 0 and ticker in forest.AI_TICKERS:
                        forest.AI_TICKERS.remove(ticker)
                        loader._ensure_ai_tickers()
                except Exception:
                    if ticker in forest.AI_TICKERS:
                        forest.AI_TICKERS.remove(ticker)
                        loader._ensure_ai_tickers()
            return

        # 3) If already shorted: return and do nothing
        if already_short:
            log_qty = int(abs_short_qty) if USE_FULL_SHARES else round(abs_short_qty, 4)
            logging.info(
                f"[{ticker}] Already short {log_qty} shares. "
                "SELL request will not modify existing short position. Skipping."
            )
            return

        # 2) No shares OWNED or SHORTED: if USE_SHORT is True, call short_shares
        if pos_qty == 0:
            if not use_shorts_enabled:
                logging.info(f"[{ticker}] No position to SELL and shorting disabled. Skipping.")
                return
            logging.info(
                f"[{ticker}] No existing position detected. "
                "USE_SHORT is True; delegating to short_shares to open a new short."
            )
            short_shares(ticker, qty, sell_price, predicted_price)
            return

    except Exception as e:
        logging.error(f"[{ticker}] Sell order failed: {e}")


def short_shares(ticker, qty, short_price, predicted_price):
    logging.info("SHORT: %s", ticker)

    if ticker.upper() in STATIC_TICKERS:
        logging.info("Ticker %s is in STATIC_TICKERS. Skipping...", ticker)
        return

    if qty <= 0:
        return

    # ✅ Batch mode with USE_TICKER_SELECTION == 2: add to waitlist CSV
    if TRADE_BATCH.active and USE_TICKER_SELECTION == 2:
        if should_add_to_waitlist(ticker, "SHORT"):
            add_to_waitlist(ticker, "SHORT", qty, short_price, predicted_price)
            logging.info("[%s] Added to waitlist CSV for SHORT", ticker)
        return
    
    # ✅ Batch mode (other modes): enqueue instead of executing
    if TRADE_BATCH.active:
        TRADE_BATCH.pending_shorts.append(PendingShort(ticker, "SHORT", qty, short_price, predicted_price, 0))
        logging.info("[%s] Queued SHORT (batch mode).", ticker)
        return

    try:
        pos = None
        try:
            pos = api.get_position(ticker)
            pos_qty = float(pos.qty)
        except Exception:
            pos = None
            pos_qty = 0.0

        already_long = pos_qty > 0
        already_short = pos_qty < 0
        abs_long_qty = pos_qty if already_long else 0.0
        abs_short_qty = abs(pos_qty) if already_short else 0.0

        if already_short:
            log_qty = int(abs_short_qty) if USE_FULL_SHARES else round(abs_short_qty, 4)
            logging.info(
                f"[{ticker}] Already short {log_qty} shares. "
                "Skipping new SHORT to prevent duplicate short position."
            )
            return

        if already_long:
            log_qty = int(abs_long_qty) if USE_FULL_SHARES else round(abs_long_qty, 4)
            logging.info(
                f"[{ticker}] Currently long {log_qty} shares. "
                "SHORT request requires a prior SELL to flatten. Skipping."
            )
            return

        if DISCORD_MODE == "on":
            send_discord_order_message(
                "SHORT", ticker, short_price, predicted_price,
                extra_info="Shorting shares via Discord bot."
            )
        else:
            account = api.get_account()
            available_cash = float(account.cash)
            total_ticker_slots = (len(forest.TICKERS) if forest.TICKERS else 0) + forest.AI_TICKER_COUNT
            total_ticker_slots = max(total_ticker_slots, 1)
            split_cash = available_cash / total_ticker_slots
            max_shares = (split_cash // short_price) if USE_FULL_SHARES else (split_cash / short_price)
            final_qty = min(qty, max_shares)

            if final_qty <= 0:
                logging.info(f"[{ticker}] Not enough split cash/margin to short any shares. Skipping.")
                return

            api.submit_order(
                symbol=ticker,
                qty=int(final_qty) if USE_FULL_SHARES else round(final_qty, 4),
                side='sell',
                type='market',
                time_in_force='gtc'
            )

            log_qty = int(final_qty) if USE_FULL_SHARES else round(final_qty, 4)
            logging.info(
                f"[{ticker}] SHORT {log_qty} at {short_price:.2f} (Predicted: {predicted_price:.2f})"
            )
            log_trade("SHORT", ticker, final_qty, short_price, predicted_price, None)
            if ticker not in forest.TICKERS and ticker not in forest.AI_TICKERS:
                forest.AI_TICKERS.append(ticker)

    except Exception as e:
        logging.error(f"[{ticker}] Short order failed: {e}")


def close_short(ticker, qty, cover_price, predicted_price):
    logging.info("COVER: %s", ticker)

    if qty <= 0:
        return

    if ticker.upper() in STATIC_TICKERS:
        logging.info("Ticker %s is in STATIC_TICKERS. Skipping...", ticker)
        return

    # ✅ Batch mode with USE_TICKER_SELECTION == 2: add to waitlist CSV
    if TRADE_BATCH.active and USE_TICKER_SELECTION == 2:
        if should_add_to_waitlist(ticker, "COVER"):
            add_to_waitlist(ticker, "COVER", qty, cover_price, predicted_price)
            logging.info("[%s] Added to waitlist CSV for COVER", ticker)
        return
    
    # ✅ Batch mode (other modes): enqueue instead of executing
    if TRADE_BATCH.active:
        TRADE_BATCH.pending_covers.append(PendingCover(ticker, "COVER", qty, cover_price, predicted_price, 0))
        logging.info("[%s] Queued COVER (batch mode).", ticker)
        return

    try:
        pos = None
        try:
            pos = api.get_position(ticker)
            pos_qty = float(pos.qty)
        except Exception:
            pos = None
            pos_qty = 0.0

        already_short = pos_qty < 0
        abs_short_qty = abs(pos_qty) if already_short else 0.0

        if not already_short or abs_short_qty <= 0:
            logging.info(f"[{ticker}] No short position to COVER. Skipping.")
            return

        coverable_qty = min(qty, abs_short_qty)
        if coverable_qty <= 0:
            logging.info(f"[{ticker}] No shares to COVER.")
            return

        if DISCORD_MODE == "on":
            send_discord_order_message(
                "COVER", ticker, cover_price, 0,
                extra_info="Covering short via Discord bot."
            )
        else:
            avg_entry = float(pos.avg_entry_price) if pos else 0.0
            api.submit_order(
                symbol=ticker,
                qty=int(coverable_qty) if USE_FULL_SHARES else round(coverable_qty, 4),
                side='buy',
                type='market',
                time_in_force='gtc'
            )
            pl = (avg_entry - cover_price) * coverable_qty
            log_qty = int(coverable_qty) if USE_FULL_SHARES else round(coverable_qty, 4)
            logging.info(
                f"[{ticker}] COVER SHORT {log_qty} at {cover_price:.2f} (P/L: {pl:.2f})"
            )
            log_trade("COVER", ticker, coverable_qty, cover_price, None, pl)
            try:
                new_pos = api.get_position(ticker)
                if float(new_pos.qty) == 0 and ticker in forest.AI_TICKERS:
                    forest.AI_TICKERS.remove(ticker)
                    loader._ensure_ai_tickers()
            except Exception:
                if ticker in forest.AI_TICKERS:
                    forest.AI_TICKERS.remove(ticker)
                    loader._ensure_ai_tickers()

    except Exception as e:
        logging.error(f"[{ticker}] Cover short failed: {e}")

# =========================
# Logging
# =========================

def log_trade(action, ticker, qty, current_price, predicted_price, profit_loss):
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row = {
        "timestamp": now_str,
        "action": action,
        "ticker": ticker,
        "quantity": qty,
        "current_price": current_price,
        "predicted_price": predicted_price,
        "profit_loss": profit_loss,
        "trade_logic": TRADE_LOGIC
    }
    df = pd.DataFrame([row])
    if not os.path.exists(TRADE_LOG_FILENAME):
        df.to_csv(TRADE_LOG_FILENAME, index=False, mode='w')
    else:
        df.to_csv(TRADE_LOG_FILENAME, index=False, mode='a', header=False)