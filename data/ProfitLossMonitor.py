# import os
# import logging
# from binance import AsyncClient
# import asyncio
# import aiohttp
# from logging import Logger
# from colorama import Fore, Style, Back
# from dotenv import load_dotenv
# import dotenv
# dotenv.load_dotenv()  # Load environment variables from a .env file


# # Correct relative imports within the data directory

# from data.TEMA_Calculations.Calculating_TripleExponentialMovingAverage import calculating_TEMA_async
# from .Fetching_and_calculating_moving_averages_and_TEMA import MovingAverageTrader


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


# class ProfitLossMonitor:
#     def __init__(self, client, symbol, api_key, api_secret):
#         self.api_key = os.getenv('BINANCE_API_KEY')
#         self.api_secret = os.getenv('BINANCE_API_SECRET')
#         self.client = client
#         self.symbol = symbol


#     async def initialize_client(self, api_key, api_secret):
#         """ Initialize Binance async client """
#         try:
#             self.client = await AsyncClient.create(api_key=api_key, api_secret=api_secret)
#             logging.info("Binance async client initialized")
#         except Exception as e:
#             logging.error(f"Error initializing Binance async client: {e}")

    # async def monitor_price_direction(client, symbol, entry_order_price):
    #     try:
    #         logging.info(f"Monitoring price direction for {symbol} from entry price: {entry_order_price}")

    #         while True:
    #             from .BinanceManagerHandler import BinanceShortManager
    #             # Fetch current price
    #             current_price = await BinanceShortManager.get_current_prices(client, symbol)
    #             if current_price is None:
    #                 logging.error(f"Failed to fetch current price for {symbol}. Retrying...")
    #                 await asyncio.sleep(1)
    #                 continue

    #             logging.info(f"Current price for {symbol}: {current_price}")

    #             # Fetch or calculate required dataframes
    #             df_7d_ma = await MovingAverageTrader.calculate_tema_and_ma_async(client, symbol, timeframe='7d')
    #             df_14d_ma = await MovingAverageTrader.calculate_tema_and_ma_async(client, symbol, timeframe='14d')
    #             df_30d_ma = await MovingAverageTrader.calculate_tema_and_ma_async(client, symbol, timeframe='30d')
    #             df_9d_tema = await calculating_TEMA_async.calculate_tema_async(client, symbol, timeframe='9d')

    #             # Evaluate conditions for activating long position
    #             if ProfitLossMonitor.should_activate_long_position(df_7d_ma, df_9d_tema, df_14d_ma, df_30d_ma, current_price):
    #                 logging.info(f"Conditions met for potential long position activation for {symbol} at price {current_price}")
    #                 # await activate_long_position(client, symbol, current_price)
    #             else:
    #                 logging.info(f"Conditions for long position activation not met for {symbol}. Continuing monitoring...")

    #             # Adjust sleep interval as per your trading strategy
    #             await asyncio.sleep(30)  # Example: Monitor every 30 seconds

    #     except Exception as e:
    #         logging.error(f"Error monitoring price direction for {symbol}: {e}", exc_info=True)

    # async def should_activate_long_position(df_7d_ma, df_9d_tema, df_14d_ma, df_30d_ma, current_price):
    #     # Implement your logic to evaluate conditions for activating a long position
    #     # Example condition: If current_price > df_7d_ma and current_price > df_14d_ma, return True
    #     return current_price > df_7d_ma and current_price > df_14d_ma

    # async def confirm_price_direction_change(self, client, symbol, entry_order_price):
    #     """
    #     Confirm if there is a significant price direction change for a given symbol.
        
    #     Args:
    #     - client: The client object to interact with the exchange API.
    #     - symbol: The trading pair symbol (e.g., 'BTCUSDT').
    #     - entry_order_price: The price at which the position was entered.

    #     Returns:
    #     - True if a significant price direction change is detected, otherwise False.
    #     """
    #     try:
    #         from .BinanceManagerHandler import BinanceShortManager
    #         from Fetching_and_calculating_moving_averages_and_TEMA import MovingAverageTrader
    #         from .candles import ProfitLossMonitor

    #         # Fetch the current price of the symbol
    #         current_price = await BinanceShortManager.get_current_prices(client, symbol)
            
    #         # Fetch historical data for the symbol
    #         historical_data = await MovingAverageTrader.fetch_historical_data(client, symbol, window=7)
            
    #         # Check if there's enough historical data (at least 3 data points) to proceed
    #         if len(historical_data) < 3:
    #             return False
            
    #         # Extract open and close prices from historical data
    #         o1, _, _, c1 = historical_data[-3]  # Previous to previous open and close
    #         _, _, _, c2 = historical_data[-2]   # Previous open and close
    #         o3, _, _, c3 = historical_data[-1]   # Latest open and close
            
    #         # Check conditions based on candlestick patterns and current price compared to entry price
    #         if (o1 > c1) and (c2 > c1) and (o3 > entry_order_price) and (current_price < entry_order_price):
    #             logging.info(f"Potential adverse price movement detected for {symbol}.")
    #             await self.monitor_unrealized_pnl(client, symbol, entry_order_price)
    #             return True
            
    #         # Fetch historical moving averages and TEMA (Triple Exponential Moving Average)
    #         ma_7days, ma_14days, ma_30days, tema_9days = await MovingAverageTrader.fetch_historical_ma(client, symbol)
            
    #         # Check conditions based on moving averages and TEMA
    #         if ma_7days and ma_14days and ma_30days and tema_9days:
    #             if (tema_9days > ma_7days) and (ma_14days > ma_7days) and (ma_30days > ma_14days):
    #                 logging.info(f"Price direction change detected for {symbol}. Starting PnL monitoring.")
    #                 await self.monitor_unrealized_pnl(client, symbol, entry_price=entry_order_price, position_size=5)
    #                 return True
    #             elif (tema_9days > ma_7days) and not (ma_14days > ma_7days) and not (ma_30days > ma_14days):
    #                 logging.info('Price direction change partially confirmed. Awaiting further signals.')
    #             elif (tema_9days > ma_7days) and (ma_14days > ma_7days) and not (ma_30days > ma_14days):
    #                 logging.info('Two signals confirmed out of three. Starting PnL monitoring.')
    #                 await self.monitor_unrealized_pnl(client, symbol, entry_price=entry_order_price, position_size=5)
    #             elif (tema_9days > ma_7days) and not (ma_14days > ma_7days) and (ma_30days > ma_14days):
    #                 logging.info('Two signals confirmed out of three. Starting PnL monitoring.')
    #                 await self.monitor_unrealized_pnl(client, symbol, entry_price=entry_order_price, position_size=5)
                
    #         # If none of the conditions are met, return False
    #         return False
        
    #     except Exception as e:
    #         logging.error(f"Error checking price direction change for {symbol}: {e}")
    #         return False

    # async def monitor_candlestick_pattern(client, symbol, unrealized_pnl_percentage):
    #     try:
    #         from .BinanceManagerHandler import BinanceShortManager
    #         from Fetching_historical_OHCL_Values import BinanceDataProcessor
    #         # Fetch recent candlestick data
    #         candlestick_data = await BinanceDataProcessor.fetch_recent_candlestick_data(client, symbol)
    #         if not candlestick_data:
    #             logging.warning(f"No recent candlestick data found for {symbol}. Cannot monitor candlestick pattern.")
    #             return

    #         # Check if imminent closure pattern detected
    #         if BinanceShortManager.is_imminent_closure_pattern_detected(candlestick_data, unrealized_pnl_percentage):
    #             logging.info(f"Imminent closure pattern detected for {symbol}. Awaiting candlestick confirmation.")

    #             # Wait for candlestick confirmation
    #             await asyncio.sleep(1)  # Adjust timeout based on candlestick interval

    #             candlestick_data = await BinanceDataProcessor.fetch_recent_candlestick_data(client, symbol)
    #             if ProfitLossMonitor.is_candlestick_confirmation_met(candlestick_data):
    #                 from .OrderManagerStopLoss import OrderManagerStopLoss
    #                 await OrderManagerStopLoss.close_position(client, symbol)
    #                 logging.info(f"Entire position closed for {symbol} based on confirmed candlestick pattern.")
    #                 return

    #     except Exception as e:
    #         logging.error(f"Error monitoring candlestick pattern for {symbol}: {e}")

    # async def is_candlestick_confirmation_met(candlestick_data):
    #     try:
    #         # Example logic to confirm candlestick pattern for closure
    #         if candlestick_data:
    #             if (float(candlestick_data[0][2]) - float(candlestick_data[1][4])) / float(candlestick_data[1][4]) > 0.0004:
    #                 return True
    #         return False

    #     except Exception as e:
    #         logging.error(f"Error confirming candlestick pattern: {e}")
    #         return False

    # async def monitor_unrealized_pnl(client, symbol, entry_price, position_size=5):
    #     try:
    #         from .candles import PnLEventEmitter, ProfitLossMonitor
            
    #         while True:
    #             # Fetch position details
    #             position_details = await ProfitLossMonitor.fetch_position_details(client, symbol)
    #             if position_details is None:
    #                 logging.warning(f"Failed to fetch position details for {symbol}.")
    #                 continue

    #             unrealized_pnl = position_details['unrealized_pnl']

    #             # Calculate PnL thresholds
    #             stop_loss_pnl = -0.05  # -5% stop loss
    #             partial_take_profit_pnl = 0.09  # 9% take profit
    #             full_take_profit_pnl = 0.10  # 10% take profit

    #             # Check if unrealized PnL reaches stop-loss threshold
    #             if unrealized_pnl <= stop_loss_pnl:
    #                 await PnLEventEmitter.pnl_event_emitter.emit_stop_loss(symbol, unrealized_pnl)
    #                 logging.info(f"Stop loss threshold reached for {symbol} at unrealized PnL: {unrealized_pnl}. Exiting position.")
    #                 break  # Exit monitoring loop and potentially exit the position

    #             # Check if unrealized PnL reaches partial take-profit threshold
    #             elif unrealized_pnl >= partial_take_profit_pnl and unrealized_pnl < full_take_profit_pnl:
    #                 await PnLEventEmitter.pnl_event_emitter.emit_partial_take_profit(symbol, unrealized_pnl)
    #                 logging.info(f"Partial take profit threshold reached for {symbol} at unrealized PnL: {unrealized_pnl}.")
    #                 # Optionally, reduce position size
    #                 position_size = position_size // 1.5

    #             # Check if unrealized PnL reaches full take-profit threshold
    #             elif unrealized_pnl >= full_take_profit_pnl:
    #                 await PnLEventEmitter.pnl_event_emitter.emit_full_take_profit(symbol, unrealized_pnl)
    #                 logging.info(f"Full take profit threshold reached for {symbol} at unrealized PnL: {unrealized_pnl}. Exiting position.")
    #                 break  # Exit monitoring loop and potentially exit the position

    #             await asyncio.sleep(1)  # Check every second

    #     except Exception as e:
    #         logging.error(f"Error monitoring unrealized PnL for {symbol}: {e}")

    # async def fetch_open_positions(client, open_positions):
    #     try:
    #         logger.info("Fetching open positions...")
    #         futures_account_trades = await client.futures_account_trades()

    #         if not futures_account_trades:
    #             logger.warning("No futures account trades found.")
    #             return

    #         for trade in futures_account_trades:
    #             try:
    #                 if trade.get('positionSide') == 'BOTH':
    #                     continue  # Skip dual positions
    #                 from .BinanceManagerHandler import BinanceShortManager

    #                 symbol = trade['symbol']
    #                 position_side = trade['positionSide']
    #                 entry_price = float(trade['entryPrice'])
    #                 stop_loss = float(trade.get('stopPrice', 0.0))
    #                 take_profit = float(trade.get('tpTriggerPrice', 0.0))

    #                 # Fetch and calculate unrealized PNL for the position
    #                 unrealized_pnl_data = await BinanceShortManager.accumulate_unrealized_pnl(client.exchange_instance, symbol, position_side)
    #                 unrealized_pnl = unrealized_pnl_data['unrealized_pnl_total'] if unrealized_pnl_data else 0.0

    #                 open_positions[symbol] = {
    #                     'position_side': position_side,
    #                     'entry_price': entry_price,
    #                     'stop_loss': stop_loss,
    #                     'take_profit': take_profit,
    #                     'unrealized_pnl': unrealized_pnl
    #                 }
    #             except KeyError as e:
    #                 logger.error(f"Error processing trade data for symbol {symbol}: {e}")
    #                 continue

    #         logger.info("Open positions fetched successfully.")

    #     except Exception as e:
    #         logger.error(f"Error fetching open positions: {e}", exc_info=True)

    # async def fetch_and_calculate_unrealized_pnl(exchange_instance, symbol, position_side='BOTH'):
    #     try:
    #         position_endpoint = "/fapi/v2/positionRisk"
    #         url = exchange_instance.base_url + position_endpoint
    #         headers = {
    #             "Content-Type": "application/json",
    #             "X-MBX-APIKEY": exchange_instance.api_key
    #         }

    #         async with aiohttp.ClientSession() as session:
    #             async with session.get(url, headers=headers) as response:
    #                 if response.status == 200:
    #                     data = await response.json()
    #                     unrealized_pnl_total = 0.0

    #                     for position in data:
    #                         if position['symbol'] == symbol:
    #                             position_side_check = position['positionSide']

    #                             # Filter positions based on position_side
    #                             if position_side == 'LONG' and position_side_check != 'LONG':
    #                                 continue
    #                             elif position_side == 'SHORT' and position_side_check != 'SHORT':
    #                                 continue

    #                             entry_price = float(position['entryPrice'])
    #                             current_price = float(position['markPrice'])
    #                             position_size = abs(float(position['positionAmt']))

    #                             # Calculate unrealized PNL for this position
    #                             if position_side_check == 'LONG':
    #                                 unrealized_pnl = (current_price - entry_price) * position_size
    #                             else:  # Assume SHORT position side
    #                                 unrealized_pnl = (entry_price - current_price) * position_size

    #                             unrealized_pnl_total += unrealized_pnl

    #                     return {
    #                         'unrealized_pnl_total': unrealized_pnl_total,
    #                         'position_data': data  # Optionally return detailed position data
    #                     }

    #                 else:
    #                     logging.error(f"Failed to fetch position information. Status code: {response.status}")
    #                     return None

    #     except Exception as e:
    #         logging.error(f"Error fetching or calculating unrealized PNL for {symbol}: {e}")
    #         return None

    # async def close(self):
    #     await self.client.close_connection()