# #This strategy is to be used for long positions signals

# import os
# import logging
# from binance.client import AsyncClient  # Import Binance client
# from utils import TradingUtils
# import asyncio
# from data.Fetching_historical_OHCL_Values import BinanceDataProcessor, KlineHandler
# from Order_handlers.OrderPlacement import OrderPlacement
# # import numpy as np
# from typing import List, Optional
# import pandas as pd


# # Load API keys from environment variables
# # api_key = os.getenv('BINANCE_API_KEY')
# # api_secret = os.getenv('BINANCE_API_SECRET')

# # client = AsyncClient.create(api_key, api_secret)

# class TradingStrategy:
#     def __init__(self, symbol: str, client: AsyncClient):
#         self.client = client
#         self.symbol = symbol
#         self.hammer_pattern_top = TradingUtils(client, symbol)
#         self.data_processor_klines_ = BinanceDataProcessor(symbol, client)
#         self.k_lines_fetching_calculations = KlineHandler(symbol)
#         self.order_placement_ = OrderPlacement(client, [symbol])

#     async def get_current_price(self) -> Optional[float]:
#         """
#         Retrieve the current price of the asset.

#         :return: The current price of the asset, or None if an error occurs.
#         """
#         try:
#             _ticker = await self.client.get_symbol_ticker(symbol=self.symbol)  # Await if this method is async
#             current_price = float(_ticker['price'])
#             logging.info(f"Current price for {self.symbol} is {current_price}")
#             return current_price
#         except Exception as e:
#             logging.error(f"Error fetching current price for {self.symbol}: {e}")
#             await asyncio.sleep(20)
#             return None

#     async def fetch_and_process_data(self, symbol: str, ohlcv_data: List[dict]):
#         # Ensure ohlcv_data is passed as klines and symbol is passed as symbol
#         df = await self.k_lines_fetching_calculations.klines_to_dataframe(ohlcv_data, symbol)
        
#         # Calculate indicators
#         df = self.calculate_7d_ma(df)
#         df = self.calculate_14d_ma(df)
#         df = self.calculate_30d_ma(df)
#         df = self.calculate_9d_tema(df)
#         df = self.calculate_30d_tema(df)
        
#         # Return relevant DataFrames for strategy
#         df_7d_ma = df[['MA_7']]
#         df_14d_ma = df[['MA_14']]
#         df_30d_ma = df[['MA_30']]
#         df_9d_tema = df[['TEMA_9']]
#         df_30d_tema = df[['TEMA_30']]
        
#         return df_7d_ma, df_9d_tema, df_14d_ma, df_30d_ma, df_30d_tema

#     async def should_activate_long_position(
#         self,
#         current_price: float,
#         df_7d_ma: Optional[pd.DataFrame],
#         df_9d_tema: Optional[pd.DataFrame],
#         df_14d_ma: Optional[pd.DataFrame],
#         df_30d_ma: Optional[pd.DataFrame],
#         candles_df: Optional[pd.DataFrame]
#     ) -> bool:
#         try:
#             # Verify that all DataFrames are provided and non-empty
#             dataframes = {
#                 'df_7d_ma': df_7d_ma,
#                 'df_9d_tema': df_9d_tema,
#                 'df_14d_ma': df_14d_ma,
#                 'df_30d_ma': df_30d_ma,
#                 'candles_df': candles_df
#             }
            
#             for df_name, df in dataframes.items():
#                 if df is None or df.empty:
#                     logging.warning(f"DataFrame {df_name} is None or empty for {self.symbol}.")
#                     return False

#             # Ensure required columns are present in the DataFrames
#             required_columns = {
#                 'df_9d_tema': ['TEMA_9'],
#                 'df_7d_ma': ['MA_7'],
#                 'df_14d_ma': ['MA_14'],
#                 'df_30d_ma': ['MA_30']
#             }

#             for df_name, columns in required_columns.items():
#                 df = dataframes[df_name]
#                 if df is None or not all(col in df.columns for col in columns):
#                     logging.warning(f"Missing columns {columns} in {df_name} for {self.symbol}.")
#                     return False

#             # Check if the last candle is a hammer pattern
#             if candles_df is None or not await self.hammer_pattern_top.is_hammer_pattern(candles_df.iloc[-1]):
#                 logging.info(f"Hammer pattern not detected for {self.symbol}. Long position will not be activated.")
#                 return False

#             # Stage 1: Basic MA and TEMA cross condition
#             if (
#                 df_9d_tema is None or df_7d_ma is None or df_14d_ma is None or df_30d_ma is None or
#                 not (df_9d_tema['TEMA_9'].iloc[-1] < df_7d_ma['MA_7'].iloc[-1] and
#                     df_7d_ma['MA_7'].iloc[-1] > df_14d_ma['MA_14'].iloc[-1] and
#                     df_14d_ma['MA_14'].iloc[-1] > df_30d_ma['MA_30'].iloc[-1])
#             ):
#                 logging.info(f"Stage 1 conditions not met for {self.symbol}. Long position will not be activated.")
#                 return False

#             # Confirm the moving average cross
#             if not await self.confirm_moving_average_cross(df_7d_ma, df_9d_tema, df_14d_ma, df_30d_ma):
#                 logging.info(f"First confirmation not met for {self.symbol}. Long position will not be activated.")
#                 return False

#             # Confirm the TEMA cross
#             if not await self.confirm_tema_cross(df_9d_tema, df_7d_ma, df_14d_ma, 9, 30):
#                 logging.info(f"Second confirmation not met for {self.symbol}. Long position will not be activated.")
#                 return False

#             # Verify TEMA is consistently above moving averages
#             if not await self.verify_tema_above_moving_averages(df_9d_tema, df_7d_ma, df_14d_ma, df_30d_ma):
#                 logging.info(f"TEMA 9 was not consistently above all MAs over the past 20-24 minutes for {self.symbol}.")
#                 return False

#             # Check historical behavior
#             if not await self.check_historical_behavior(df_9d_tema, df_7d_ma, df_14d_ma, df_30d_ma, df_30d_tema=df):
#                 logging.info(f"Historical behavior not met for {self.symbol}. Long position will not be activated.")
#                 return False

#             logging.info(f"\033[42mFinal confirmation triggered. Long position can be activated for {self.symbol}.\033[0m")
#             return True

#         except KeyError as e:
#             logging.error(f"KeyError occurred while checking conditions to activate long position for {self.symbol}: {e}")
#             await asyncio.sleep(20)
#             return False
#         except IndexError as e:
#             logging.error(f"IndexError occurred while checking conditions to activate long position for {self.symbol}: {e}")
#             await asyncio.sleep(20)
#             return False
#         except Exception as e:
#             logging.error(f"Unexpected error while checking conditions to activate long position for {self.symbol}: {e}")
#             await asyncio.sleep(20)
#         return False
        
#     async def verify_tema_above_moving_averages(self, df_9d_tema, df_7d_ma, df_14d_ma, df_30d_ma) -> bool:
#         if any(df.empty for df in [df_9d_tema, df_7d_ma, df_14d_ma, df_30d_ma]):
#             logging.warning(f"One or more DataFrames are empty or have insufficient data.")
#             return False

#         num_minutes = 20
#         for i in range(1, num_minutes + 1):
#             try:
#                 if not (
#                     df_9d_tema['TEMA_9'].iloc[-i] > df_7d_ma['MA_7'].iloc[-i] and
#                     df_9d_tema['TEMA_9'].iloc[-i] > df_14d_ma['MA_14'].iloc[-i] and
#                     df_9d_tema['TEMA_9'].iloc[-i] > df_30d_ma['MA_30'].iloc[-i]
#                 ):
#                     return False
#             except IndexError:
#                 logging.warning(f"Not enough data for verification. Exiting the check at minute {i}.")
#                 return False

#         return True


    
#     def calculate_7d_ma(self, df: pd.DataFrame) -> pd.DataFrame:
#         df['MA_7'] = df['close'].rolling(window=7).mean()
#         return df

#     def calculate_14d_ma(self, df: pd.DataFrame) -> pd.DataFrame:
#         df['MA_14'] = df['close'].rolling(window=14).mean()
#         return df
    
#     def calculate_30d_ma(self, df: pd.DataFrame) -> pd.DataFrame:
#         df['MA_30'] = df['close'].rolling(window=30).mean()
#         return df
    
#     def calculate_ema(self, series: pd.Series, period: int) -> pd.Series:
#         return series.ewm(span=period, adjust=False).mean()


#     def calculate_9d_tema(self, df: pd.DataFrame) -> pd.DataFrame:
#         df['EMA_9'] = self.calculate_ema(df['close'], 9)  # Pass the 'close' column (Series)
#         df['EMA_9_2'] = self.calculate_ema(df['EMA_9'], 9)
#         df['EMA_9_3'] = self.calculate_ema(df['EMA_9_2'], 9)
#         df['TEMA_9'] = 3 * (df['EMA_9'] - df['EMA_9_2']) + df['EMA_9_3']
#         return df
    
#     def calculate_30d_tema(self, df: pd.DataFrame) -> pd.DataFrame:
#         df['EMA_30'] = self.calculate_ema(df['close'], 30)  # Pass the 'close' column (Series)
#         df['EMA_30_2'] = self.calculate_ema(df['EMA_30'], 30)
#         df['EMA_30_3'] = self.calculate_ema(df['EMA_30_2'], 30)
#         df['TEMA_30'] = 3 * (df['EMA_30'] - df['EMA_30_2']) + df['EMA_30_3']
#         return df

#     async def check_historical_behavior(self, df_9d_tema: pd.DataFrame, df_7d_ma: pd.DataFrame, df_14d_ma: pd.DataFrame, df_30d_ma: pd.DataFrame, df_30d_tema: pd.DataFrame) -> bool:
#         """
#         Check if TEMA 9 was below all MAs before it started to cross above them.

#         :param df_9d_tema: DataFrame containing the 9-day TEMA.
#         :param df_7d_ma: DataFrame containing the 7-day moving average.
#         :param df_14d_ma: DataFrame containing the 14-day moving average.
#         :param df_30d_ma: DataFrame containing the 30-day moving average.
#         :param df_30d_tema: DataFrame containing the 30-day TEMA.
#         :return: True if TEMA 9 was below all MAs previously, False otherwise.
#         """
#         historical_window = 24  # Minutes to check historical behavior

#         # Check that all DataFrames have enough data
#         if len(df_9d_tema) < historical_window or len(df_7d_ma) < historical_window or len(df_14d_ma) < historical_window or len(df_30d_ma) < historical_window or len(df_30d_tema) < historical_window:
#             logging.warning("Not enough data to perform historical behavior check.")
#             return False

#         for i in range(1, historical_window + 1):  # Check the past 24 periods
#             if (
#                 df_9d_tema['TEMA_9'].iloc[-i] >= df_7d_ma['MA_7'].iloc[-i] or
#                 df_9d_tema['TEMA_9'].iloc[-i] >= df_14d_ma['MA_14'].iloc[-i] or
#                 df_9d_tema['TEMA_9'].iloc[-i] >= df_30d_ma['MA_30'].iloc[-i] or
#                 df_9d_tema['TEMA_9'].iloc[-i] >= df_30d_tema['TEMA_30'].iloc[-i]
#             ):
#                 logging.info(f"TEMA 9 was not consistently below all MAs in the past {historical_window} periods for {self.symbol}.")
#                 return False
#             await asyncio.sleep(1)  # Simulate waiting for real-time data

#         return True

#     async def confirm_moving_average_cross(self, df_7d_ma: pd.DataFrame, df_9d_tema: pd.DataFrame, df_14d_ma: pd.DataFrame, df_30d_ma: pd.DataFrame) -> bool:
#         try:
#             num_checks = 100
#             check_interval = 1

#             for _ in range(num_checks):
#                 # Check if any DataFrame is empty
#                 if any(df.empty for df in [df_7d_ma, df_9d_tema, df_14d_ma, df_30d_ma]):
#                     logging.warning(f"One or more DataFrames are empty during MA cross confirmation for {self.symbol}.")
#                     return False

#                 # Access the last element of each DataFrame
#                 current_7d_ma = df_7d_ma['MA_7'].iloc[-1]
#                 current_9d_tema = df_9d_tema['TEMA_9'].iloc[-1]
#                 current_14d_ma = df_14d_ma['MA_14'].iloc[-1]
#                 current_30d_ma = df_30d_ma['MA_30'].iloc[-1]

#                 # Check moving average cross condition
#                 if current_9d_tema < current_7d_ma and current_7d_ma > current_14d_ma and current_14d_ma > current_30d_ma:
#                     logging.info(f"Moving average cross confirmed for {self.symbol}.")
#                     return True

#                 # Wait before the next check
#                 await asyncio.sleep(check_interval)

#             logging.warning(f"Moving average cross confirmation failed after {num_checks} checks for {self.symbol}.")
#             return False

#         except Exception as e:
#             logging.error(f"Error during moving average cross confirmation for {self.symbol}: {e}")
#             await asyncio.sleep(20)  # Optionally wait before retrying
#             return False

#     async def confirm_tema_cross(
#         self,
#         df_9d_tema: pd.DataFrame,
#         df_7d_ma: pd.DataFrame,
#         df_14d_ma: pd.DataFrame,
#         tema_short_period: int,
#         tema_long_period: int
#     ) -> bool:
#         try:
#             num_checks = 100
#             check_interval = 1

#             # Check if any DataFrame is empty
#             if any(df.empty for df in [df_9d_tema, df_7d_ma, df_14d_ma]):
#                 logging.warning(f"One or more DataFrames are empty during TEMA cross confirmation for {self.symbol}.")
#                 return False

#             for _ in range(num_checks):
#                 # Access the latest TEMA value
#                 current_9d_tema = df_9d_tema['TEMA_9'].iloc[-1]
#                 latest_ma_7 = df_7d_ma['MA_7'].iloc[-1]
#                 latest_ma_14 = df_14d_ma['MA_14'].iloc[-1]

#                 # Check TEMA cross condition
#                 if (
#                     current_9d_tema > latest_ma_7 and
#                     current_9d_tema > latest_ma_14
#                 ):
#                     logging.info(f"TEMA cross confirmed for {self.symbol}. TEMA_9: {current_9d_tema}, MA_7: {latest_ma_7}, MA_14: {latest_ma_14}.")
#                     return True

#                 await asyncio.sleep(check_interval)

#             logging.warning(f"TEMA cross confirmation failed after {num_checks} checks for {self.symbol}.")
#             return False

#         except Exception as e:
#             logging.error(f"Error during TEMA cross confirmation for {self.symbol}: {e}")
#             await asyncio.sleep(20)  # Optionally wait before retrying
#             return False

#     # async def activate_long_position(self, current_price):
#     #     """
#     #     Activate a long position by placing a market buy order.

#     #     :param current_price: The current price of the asset.
#     #     :return: Response from the order placement API call.
#     #     """
#     #     try:
#     #         logging.info(f"Activating long position for {self.symbol} at current price {current_price}")

#     #         order_response = await self.client.futures_create_order(
#     #             symbol=self.symbol,
#     #             side='BUY',
#     #             type='MARKET',
#     #             # quantity=self.quantity,
#     #         )

#     #         if 'orderId' in order_response:
#     #             logging.info(f"Long position opened successfully for {self.symbol}: {order_response}")
#     #         else:
#     #             logging.error(f"Failed to open long position for {self.symbol}: {order_response}")
#     #             await asyncio.sleep(20)

#     #         return order_response

#     #     except Exception as e:
#     #         logging.error(f"Error activating long position for {self.symbol}: {e}")
#     #         await asyncio.sleep(20)
#     #         return None

#     # async def set_take_profit_and_stop_loss(self, entry_price, take_profit_ratio=1.02, stop_loss_ratio=0.98):
#     #     """
#     #     Set take profit and stop loss levels after opening a position by delegating the task to the
#     #     existing class responsible for handling take profit and stop loss.

#     #     :param entry_price: The price at which the long position was opened.
#     #     :param take_profit_ratio: The ratio to calculate the take profit price (e.g., 1.02 for 2% profit).
#     #     :param stop_loss_ratio: The ratio to calculate the stop loss price (e.g., 0.98 for 2% loss).
#     #     :return: Response from the take profit and stop loss order placement API calls.
#     #     """
#     #     try:
#     #         # Delegate to the other class responsible for setting take profit and stop loss
#     #         take_profit_price = entry_price * take_profit_ratio
#     #         stop_loss_price = entry_price * stop_loss_ratio

#     #         logging.info(f"Delegating take profit and stop loss configuration for {self.symbol}.")

#     #         # Assuming the other class has a similar method to handle this functionality
#     #         order_response = await self.other_class_instance.set_take_profit_and_stop_loss(
#     #             symbol=self.symbol, 
#     #             take_profit_price=take_profit_price, 
#     #             stop_loss_price=stop_loss_price
#     #         )

#     #         logging.info(f"Take profit and stop loss configured for {self.symbol}: {order_response}")
#     #         return order_response

#     #     except Exception as e:
#     #         logging.error(f"Error setting take profit and stop loss for {self.symbol}: {e}")
#     #         await asyncio.sleep(20)
#     #         return None



#     # async def execute_strategy(self, df_7d_ma, df_9d_tema, df_14d_ma, df_30d_ma, candles_df):
#     #     """
#     #     Execute the entire strategy: check conditions, open a position, and set take profit/stop loss.

#     #     :param df_7d_ma: DataFrame containing the 7-day moving average.
#     #     :param df_9d_tema: DataFrame containing the 9-day TEMA.
#     #     :param df_14d_ma: DataFrame containing the 14-day moving average.
#     #     :param df_30d_ma: DataFrame containing the 30-day moving average.
#     #     :param candles_df: DataFrame containing OHLC data for candlestick patterns.
#     #     """
#     #     current_price = await self.get_current_price()

#     #     # Ensure current_price is valid
#     #     if current_price is None or not isinstance(current_price, float):
#     #         logging.error(f"Invalid current price: {current_price}. Cannot execute strategy.")
#     #         await asyncio.sleep(20)
#     #         return

#     #     # Example: Calculate quantity based on balance or use a fixed value
#     #     quantity = await self.calculate_quantity(current_price)

#     #     # Ensure quantity is valid
#     #     if quantity <= 0:
#     #         logging.error(f"Invalid quantity: {quantity}. Cannot execute strategy.")
#     #         await asyncio.sleep(20)
#     #         return

#     #     # Check if conditions to activate long position are met
#     #     if await self.should_activate_long_position(current_price, df_7d_ma, df_9d_tema, df_14d_ma, df_30d_ma, candles_df):
#     #         entry_price = current_price  # Or use the value from the API

#     #         # Open long position using the OrderPlacement class
#     #         order_response = await self.order_placement_.open_long_position(symbol=self.symbol, entry_price=entry_price, quantity=quantity)

#     #         if order_response:
#     #             entry_price = float(order_response['fills'][0]['price'])
#     #             await self.order_placement_.set_take_profit_and_stop_loss(entry_price)
#     #         else:
#     #             logging.error(f"Failed to place the order for {self.symbol}.")
#     #             await asyncio.sleep(20)

#     async def close(self):
#             """Close the Binance client session."""
#             if self.client:
#                 await self.client.close_connection()
#                 logging.info("Binance client connection closed.")
# # def create_dummy_df(num_rows):
# #     df = pd.DataFrame({
# #         'TEMA': np.random.randn(num_rows).cumsum(),
# #         'MA_7': np.random.randn(num_rows).cumsum(),
# #         'MA_14': np.random.randn(num_rows).cumsum(),
# #         'MA_30': np.random.randn(num_rows).cumsum(),
# #     })
# #     return df


# # # Define the trading symbol
# # symbol = 'BTCUSDT'

# # # # Initialize the Binance client with API key and secret
# # client = Client(api_key, api_secret)

# # # # Create an instance of TradingStrategy
# # strategy = TradingStrategy(symbol='BTCUSDT', client=client)

# #      async def run_strategy():
# #         df_7d_ma = create_dummy_df(50)
# #         df_9d_tema = create_dummy_df(50)
# #         df_14d_ma = create_dummy_df(50)
# #         df_30d_ma = create_dummy_df(50)
# #         candles_df = create_dummy_df(50)
    
# #         await strategy.execute_strategy(df_7d_ma, df_9d_tema, df_14d_ma, df_30d_ma, candles_df)

# # if __name__ == "__main__":
# #     asyncio.run(run_strategy())