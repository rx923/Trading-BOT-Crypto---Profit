# import os
# import logging
# from binance.client import Client, AsyncClient
# from logging import Logger
# import asyncio
# from colorama import Fore, Style, Back
# import datetime
# from BinanceManagerHandler import BinanceShortManager
# from OrderManagerStopLoss import OrderManagerStopLoss
# from data.AccountManager.AccountManager import AccountManager
# # from ProfitLossMonitor import ProfitLossMonitor
# from OrderManagerTakeProfit import OrderManagerTakeProfit
# from Order_handlers.OrderPlacement import OrderPlacement

# class ColoredFormatter(logging.Formatter):
#     def format(self, record):
#         # Set colors based on log level
#         if record.levelno == logging.INFO:
#             color = Fore.GREEN
#         elif record.levelno == logging.WARNING:
#             color = Fore.YELLOW
#         elif record.levelno == logging.ERROR:
#             color = Fore.RED
#         else:
#             color = Style.RESET_ALL

#         # Format the message with color
#         formatted_message = f"{color}{record.levelname} - {record.getMessage()}{Style.RESET_ALL}"

#         return formatted_message

# # Configure the root logger
# logger = logging.getLogger()
# logger.setLevel(logging.DEBUG)
# handler = logging.StreamHandler()
# handler.setLevel(logging.DEBUG)
# formatter = ColoredFormatter()
# handler.setFormatter(formatter)
# logger.addHandler(handler)


# class ManagingShortPositions:
#     def __init__(self):
#         self.api_key = os.getenv('BINANCE_API_KEY')
#         self.api_secret = os.getenv('BINANCE_API_SECRET')
#         self.client = None
#         self.binance_manager = BinanceShortManager(self.client)

#     async def initialize_client(self, api_key, api_secret):
#         """ Initialize Binance async client """
#         try:
#             self.client = await AsyncClient.create(api_key=api_key, api_secret=api_secret)
#             logging.info("Binance async client initialized")
#         except Exception as e:
#             logging.error(f"Error initializing Binance async client: {e}")

#     async def manage_short_positions(self, symbol):
#         try:
#             logger.info(f"Managing short positions for {symbol}")

#             # Initialize variables and flags
#             current_candle_close = None
#             previous_candle_close = None
#             active_position = False

#             # Dictionary to store open positions
#             open_positions = {}

#             while True:
#                 # Fetch necessary data for evaluation
#                 window = 100  # Set the window size for historical data
#                 historical_data = await self.binance_manager.fetch_historical_data(self.client, symbol, window=7)
#                 if historical_data is not None:
#                     ma_7days = historical_data['open'].rolling(window=7).mean().iloc[-1]
#                     ma_14days = historical_data['open'].rolling(window=14).mean().iloc[-1]
#                     ma_30days = historical_data['open'].rolling(window=30).mean().iloc[-1]
#                     tema_9days = await self.binance_manager.calculate_tema_async(historical_data)
#                     current_candle_close = historical_data['close'].iloc[-1]

#                     # Default previous_candle_close if it's None
#                     if previous_candle_close is None:
#                         previous_candle_close = current_candle_close

#                 # Fetch open and ongoing positions
#                 await ProfitLossMonitor.fetch_open_positions(self.client, open_positions)
#                 await self.binance_manager.check_ongoing_positions(open_positions)

#                 # Check if there is an ongoing short position for the symbol
#                 position_info = open_positions.get(symbol)

#                 if position_info:
#                     logger.info(f"Short position already open for {symbol}. Monitoring...")

#                     # Handle active position management if not already managed
#                     if not active_position:
#                         await self.manage_short_positions(symbol, action='open')
#                         active_position = True

#                     # Fetch the latest unrealized PnL for the open position
#                     unrealized_pnl = position_info.get('unrealized_pnl')

#                     if unrealized_pnl is not None:
#                         logger.info(f"{symbol} - Unrealized PnL: {unrealized_pnl}")

#                         # Dynamic risk level and take profit ratio
#                         risk_level = 0.03  # Example: 3%
#                         take_profit_ratio = 1.5  # Example: 1.5x the risk level for take profit

#                         stop_loss_price, take_profit_price = OrderManagerStopLoss.calculate_dynamic_stop_loss(
#                             current_candle_close, risk_level, take_profit_ratio
#                         )

#                         logger.info(f"Stop Loss Price: {stop_loss_price}, Take Profit Price: {take_profit_price}")

#                         # Monitor candlestick patterns and confirm closures
#                         # await YourTradingBot.monitor_candlestick_pattern(self.client, symbol, unrealized_pnl)

#                         # Handle partial profit-taking based on conditions
#                         if current_candle_close < previous_candle_close and unrealized_pnl > 0:
#                             logger.info(f"Taking partial profit for {symbol}: Current candle closed lower than previous candle and unrealized PnL is positive")
#                             await OrderManagerStopLoss.close_position(self.client, symbol, 0.5)

#                         # Check if take profit orders are placed
#                         if not position_info.get('take_profit_orders_placed', False):
#                             await OrderManagerTakeProfit.place_take_profit_order(self.client, symbol, position_info, unrealized_pnl, current_candle_close, current_price=self.binance_manager.get_current_prices, is_short=True)

#                         # Additional conditions for managing the position
#                         elif (tema_9days > current_candle_close > previous_candle_close) or (
#                             tema_9days > ma_7days and tema_9days > ma_14days and tema_9days > ma_30days):
#                             logger.info(f"Closing short position for {symbol}: TEMA crossed above previous candle's close value or crossed above MA thresholds")
#                             await OrderManagerStopLoss.close_position(self.client, symbol)
#                             await self.manage_short_positions(symbol, action='close')
#                             break

#                         else:
#                             logger.info(f"Monitoring {symbol}: Indicator values did not trigger actions")

#                     else:
#                         logger.error(f"Failed to fetch unrealized PnL for {symbol}")

#                 else:
#                     logger.info(f"No open short position found for {symbol}. Checking conditions to open...")

#                     # Check account balance before opening a new position
#                     usdt_balance, _ = await AccountManager.fetch_account_balance(symbol)
#                     if usdt_balance is None or usdt_balance < 5:
#                         logger.warning(f"Insufficient USDT balance to open or monitor short position for {symbol}. Current balance: {usdt_balance}")
#                         await asyncio.sleep(1)  # Wait for a minute before checking again
#                         continue

#                     # Open short position if conditions are met
#                     position_size_increase = 1  # Example: adjust based on logic
#                     confirmation = await OrderPlacement.open_short_position(self.client, symbol, current_candle_close, usdt_balance, position_size_increase)
#                     if confirmation:
#                         logger.info(f"Short position opened successfully for {symbol}")
                    
#                 await asyncio.sleep(1)

#         except Exception as e:
#             logger.error(f"Error managing short positions for {symbol}: {e}")

#     async def check_short_position(client, symbols, timeframe, limit, account_balance):
#         logging.info(f"Starting short position check for symbols: {symbols}")

#         async def process_symbol(symbol):
#             try:
#                 logging.info(f"Checking {symbol} for short position.")

#                 # Fetch current price
#                 current_price = await BinanceShortManager.get_current_prices(client, symbol)
#                 if current_price is None:
#                     logging.error(f"Failed to fetch current price for {symbol}. Skipping.")
#                     return

#                 # Fetch OHLCV data for 1m timeframe
#                 df_1m = await BinanceShortManager.fetch_ohlcv_async(client, symbol, timeframe, limit)
#                 if df_1m is None or df_1m.empty:
#                     logging.warning(f"{symbol} - No OHLCV data or empty DataFrame in {timeframe} timeframe.")
#                     return

#                 # Fetch OHLCV data for 3m and 5m timeframes asynchronously
#                 df_3m = await BinanceShortManager.fetch_ohlcv_async(client, symbol, '3m', limit)
#                 df_5m = await BinanceShortManager.fetch_ohlcv_async(client, symbol, '5m', limit)

#                 # Check if data is valid
#                 if df_3m is None or df_3m.empty or df_5m is None or df_5m.empty:
#                     logging.warning(f"{symbol} - No valid data for 3m or 5m timeframes.")
#                     return

#                 # Calculate moving averages and TEMA for 1m timeframe if conditions met
#                 should_open, prev_close, curr_close, tema_value = await BinanceShortManager.should_open_short_position(
#                     client, symbol, df_1m=df_1m, df_3m=df_3m, df_5m=df_5m
#                 )

#                 if should_open:
#                     current_time = datetime.now().strftime("%H:%M:%S")
#                     logging.info(f"{symbol} - {current_time} - Primary condition met for potential short position. Waiting for confirmation.")

#                     # Check if conditions allow doubling the position size
#                     should_double_position = await ManagingShortPositions.check_short_position_conditions(
#                         client, symbol, df_1m=df_1m, df_3m=df_3m, df_5m=df_5m
#                     )

#                     # Attempt to fetch account balance
#                     usdt_balance = await AccountManager.fetch_account_balance(symbol)
#                     if usdt_balance <= 0:
#                         logging.error("Insufficient or zero USDT balance. Skipping order.")
#                         return

#                     logging.info(f"Account balance fetched. USDT balance: {usdt_balance}")

#                     # Set position size increase based on conditions
#                     position_size_increase = 1  # Default to 1 (no increase)
#                     if should_double_position:
#                         position_size_increase = 2

#                     # Open short position with increased size if confirmed
#                     confirmation = await OrderPlacement.open_short_position(
#                         client, symbol, curr_close, account_balance, position_size_increase
#                     )
#                     if confirmation:
#                         # Log confirmation details including prev_close, curr_close, and tema_value
#                         logging.info(
#                             f"{symbol} - Confirmed condition met for short position: "
#                             f"Previous close = {prev_close}, Current close = {curr_close}, TEMA = {tema_value}."
#                         )

#             except Exception as e:
#                 logging.error(f"Error in check_short_position for {symbol}: {e}")

#         # Process each symbol asynchronously
#         await asyncio.gather(*(process_symbol(symbol) for symbol in symbols))

#     async def check_short_position_conditions(client, symbol, df_1m=None, df_3m=None, df_5m=None):
#         try:
#             if df_1m is None or df_3m is None or df_5m is None:
#                 logging.warning(f"Dataframes are not provided for {symbol}. Skipping condition check.")
#                 return False

#             # Implement your conditions for checking if the position should be doubled
#             should_open, prev_close, curr_close, tema_value = await BinanceShortManager.should_open_short_position(
#                 client, symbol, df_1m=df_1m, df_3m=df_3m, df_5m=df_5m
#             )

#             if should_open:
#                 # Example of using the client parameter
#                 account_info = await client.futures_account()
#                 # Process account_info if needed
#                 print(account_info)

#                 # Log confirmation details including prev_close, curr_close, and tema_value
#                 logging.info(
#                     f"{symbol} - Confirmed condition met for short position: "
#                     f"Previous close = {prev_close}, Current close = {curr_close}, TEMA = {tema_value}."
#                 )

#                 return True

#             return False

#         except Exception as e:
#             logging.error(f"Error checking short position conditions for {symbol}: {e}")
#             return False

#     async def close(self):
#         await self.client.close_connection()