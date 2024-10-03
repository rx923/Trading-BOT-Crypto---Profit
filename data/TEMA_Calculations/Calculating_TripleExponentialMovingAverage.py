# # File: Calculating_TripleExponentialMovingAverage.py

# import logging
# import pandas as pd
# import asyncio
# import aiodns
# import dotenv
# from binance.client import Client, AsyncClient
# from colorama import Fore, Style, Back
# import os
# from typing import Optional, Tuple, List, Any
# from data.Fetching_historical_OHCL_Values import BinanceDataProcessor
# # from data.Fetching_and_calculating_moving_averages_and_TEMA import MovingAverageTrader
# from utils import TradingUtils
# # Set the event loop policy to SelectorEventLoop for compatibility with aiodns
# if os.name == 'nt':  # Check if the OS is Windows
#     asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


# # Initialize the Binance client with appropriate API keys
# api_key = os.getenv('BINANCE_API_KEY')
# api_secret = os.getenv('BINANCE_API_SECRET')
# client = AsyncClient(api_key, api_secret)

# # Set the event loop policy to SelectorEventLoop for compatibility with aiodns
# if os.name == 'nt':  # Check if the OS is Windows
#     asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# dotenv.load_dotenv()  # Load environment variables from a .env file

# logging.basicConfig(level=logging.INFO, level_1=logging.WARNING, level_2=logging.ERROR)

# # Initialize BinanceDataProcessor


# class ColoredFormatter(logging.Formatter):
#     """
#     Formatter to apply colored formatting to log messages based on their level.
#     """
#     def format(self, record):
#         color = {
#             logging.INFO: Fore.GREEN,
#             logging.WARNING: Fore.YELLOW,
#             logging.ERROR: Fore.RED
#         }.get(record.levelno, Style.RESET_ALL)

#         if hasattr(record, 'blue_background') and record.blue_background:
#             color = Back.BLUE + Fore.WHITE

#         formatted_message = f"{color}{record.levelname} - {record.getMessage()}{Style.RESET_ALL}"
#         return formatted_message


# # Configure the root custom_logger 
# custom_logger  = logging.getLogger()
# custom_logger .setLevel(logging.DEBUG)
# handler = logging.StreamHandler()
# handler.setLevel(logging.DEBUG)
# formatter = ColoredFormatter()
# handler.setFormatter(formatter)
# custom_logger .addHandler(handler)


# class CustomLoggerAdapter(logging.LoggerAdapter):
#     def process(self, msg, kwargs):
#         extra = self.extra.copy()
#         if 'extra' in kwargs:
#             extra.update(kwargs['extra'])
#         kwargs['extra'] = extra
#         return msg, kwargs
# # Create an instance of LoggerAdapter with the custom attribute
# custom_logger = CustomLoggerAdapter(custom_logger , {'blue_background': True})

# class calculating_TEMA_async:
#     def __init__(self, symbol, client, api_key=None, api_secret=None, window_9: int = 9, window_30: int = 30):
#         self.symbol = symbol
#         self.client = client
#         self.api_key = api_key
#         self.api_secret = api_secret
#         self.window_9 = window_9
#         self.window_30 = window_30
#         self.trader = BinanceDataProcessor(client, api_key, api_secret)
#         self.logger = logging.getLogger('calculating_TEMA_async')

#         if not isinstance(self.trader, BinanceDataProcessor):
#             self.logger.error(f"Invalid trader instance: {type(self.trader)}")
#             raise ValueError("Trader instance must be provided and should be of type BinanceDataProcessor.")
        
#         # Initialize MovingAverageTrader with client and window sizes
#         # self.moving_average_trader = MovingAverageTrader(client, window_9, window_30)  

#     # def process_data(self, symbol, data: List[List[Any]]) -> pd.DataFrame:
#     #     if data:
#     #         try:
#     #             if isinstance(data, list) and isinstance(data[0], list):
#     #                 df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
#     #             elif isinstance(data, pd.DataFrame):
#     #                 df = data
#     #             else:
#     #                 self.logger.error(f"Unsupported data format for {symbol}: {type(data)}")
#     #                 return pd.DataFrame() 

#     #             # df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
#     #             df.set_index('timestamp', inplace=True)
                
#     #             self.logger.debug(f"DataFrame shape for {symbol}: {df.shape}")
#     #             self.logger.debug(f"DataFrame head for {symbol}:\n{df.head()}")

#     #             # is_valid, validation_message = TradingUtils.validate_data(df)
#     #             # if is_valid:
#     #                 # self.logger.info(f"Successfully validated data for {symbol}.")
#     #                 # return df
#     #             # else:
#     #                 # self.logger.error(f"Validation failed for {symbol}: {validation_message}")
#     #                 # return pd.DataFrame() 

#     #         except Exception as e:
#     #             self.logger.error(f"Error converting data to DataFrame for {symbol}: {str(e)}")
#     #             self.logger.debug(e, exc_info=True)
#     #             return pd.DataFrame() 

#     #     else:
#     #         self.logger.error(f"No data to process for {symbol}.")
#     #         return pd.DataFrame() 

#     async def calculate_tema_async(self, symbol, _ohlcv_df, ma_condition=None) -> bool:
#         retry_attempts = 10
#         retry_delay = 5

#         for attempt in range(retry_attempts):
#             try:
#                 self.logger.info(f"Attempt {attempt + 1} to calculate TEMA for {symbol} with {ma_condition}")

#                 # Check DataFrame validity
#                 if _ohlcv_df.empty or 'close' not in _ohlcv_df.columns:
#                     self.logger.error(f"DataFrame is empty or missing 'close' column for {symbol}. Skipping TEMA calculation.")
#                     await asyncio.sleep(retry_delay)
#                     continue

#                 # Process the DataFrame
#                 _ohlcv_df['close'] = pd.to_numeric(_ohlcv_df['close'], errors='coerce')
#                 _ohlcv_df = _ohlcv_df.dropna(subset=['close'])
#                 if _ohlcv_df['close'].isnull().all():
#                     self.logger.error(f"All 'close' values are NaN for {symbol}. Skipping TEMA calculation.")
#                     await asyncio.sleep(retry_delay)
#                     continue

#                 # Calculate TEMA values
#                 _ohlcv_df['ema_9'] = _ohlcv_df['close'].ewm(span=self.window_9, adjust=False).mean()
#                 _ohlcv_df['ema_ema_9'] = _ohlcv_df['ema_9'].ewm(span=self.window_9, adjust=False).mean()
#                 _ohlcv_df['ema_ema_ema_9'] = _ohlcv_df['ema_ema_9'].ewm(span=self.window_9, adjust=False).mean()
#                 _ohlcv_df['TEMA_9'] = 3 * (_ohlcv_df['ema_9'] - _ohlcv_df['ema_ema_9']) + _ohlcv_df['ema_ema_ema_9']
#                 tema_9_value = _ohlcv_df['TEMA_9'].iloc[-1]

#                 _ohlcv_df['ema_30'] = _ohlcv_df['close'].ewm(span=self.window_30, adjust=False).mean()
#                 _ohlcv_df['ema_ema_30'] = _ohlcv_df['ema_30'].ewm(span=self.window_30, adjust=False).mean()
#                 _ohlcv_df['ema_ema_ema_30'] = _ohlcv_df['ema_ema_30'].ewm(span=self.window_30, adjust=False).mean()
#                 _ohlcv_df['TEMA_30'] = 3 * (_ohlcv_df['ema_30'] - _ohlcv_df['ema_ema_30']) + _ohlcv_df['ema_ema_ema_30']
#                 tema_30_value = _ohlcv_df['TEMA_30'].iloc[-1]

#                 # Log TEMA values
#                 self.logger.info(f"TEMA calculation completed for {symbol}. TEMA_9={tema_9_value}, TEMA_30={tema_30_value}")
#                 return True  # Indicate success

#             except AttributeError as e:
#                 self.logger.error(f"AttributeError in calculating TEMA for {symbol}: {str(e)}")
#                 self.logger.debug(e, exc_info=True)
#                 await asyncio.sleep(retry_delay)

#             except Exception as e:
#                 self.logger.error(f"Unexpected error in calculating TEMA for {symbol}: {str(e)}")
#                 self.logger.debug(e, exc_info=True)
#                 await asyncio.sleep(retry_delay)

#             finally:
#                 self.logger.info(f"Completed processing for symbol {symbol}. Attempt {attempt + 1}")

#         # If all attempts fail, log and return False
#         self.logger.error(f"Max retry attempts reached for {symbol}. Aborting TEMA calculation.")
#         return False

# # async def main():
# #     tema_calculator = calculating_TEMA_async()
# #     await tema_calculator.initialize_client()
# #     # Example usage
# #     await tema_calculator.calculate_tema_async('BTCUSDT', 'short_signal')
# #     # Close the client at the end
# #     await tema_calculator.close_client()
# # if __name__ == "__main__":
# #     asyncio.run(main())