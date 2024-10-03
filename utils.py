# utils.py
import os
import json
import pandas as pd
import logging
from binance.client import AsyncClient, Client
# from data.Fetching_and_calculating_moving_averages_and_TEMA import MovingAverageTrader

from typing import Optional, Tuple, List, Any
from dotenv import load_dotenv
import dotenv
import asyncio
# from data.Fetching_historical_OHCL_Values import BinanceDataProcessor

from data.AccountManager import AccountManager

dotenv.load_dotenv()  # Load environment variables from a .env file

# Load API credentials from environment variables
# api_key = os.getenv('BINANCE_API_KEY')
# api_secret = os.getenv('BINANCE_API_SECRET')
# client = AsyncClient(api_key=api_key, api_secret=api_secret)



class TradingUtils:
    def __init__(self, client, symbol):
        self.client = client
        self.symbol = symbol
        self.api_key = os.getenv('BINANCE_API_KEY')
        self.api_secret = os.getenv('BINANCE_API_SECRET')
        # self.fetching_klines = BinanceDataProcessor(symbol, client)

    # @staticmethod
    async def is_hammer_pattern(self, candle):
        try:
            open_price = float(candle['open'])
            high_price = float(candle['high'])
            low_price = float(candle['low'])
            close_price = float(candle['close'])

            body_size = abs(close_price - open_price)
            lower_shadow_size = open_price - low_price if open_price > close_price else close_price - low_price
            upper_shadow_size = high_price - open_price if open_price > close_price else high_price - close_price

            is_hammer = (
                close_price > open_price and
                lower_shadow_size >= 2 * body_size and
                upper_shadow_size <= 0.5 * body_size and
                (close_price - open_price) / open_price >= 0.0025
            )

            is_inverted_hammer = (
                close_price > open_price and
                upper_shadow_size >= 2 * body_size and
                lower_shadow_size <= 0.5 * body_size and
                (close_price - open_price) / open_price >= 0.0025
            )

            if is_hammer or is_inverted_hammer:
                logging.info("Hammer pattern detected.")
                return True
            else:
                logging.info("No hammer pattern detected.")
                return False
        except Exception as e:
            logging.error(f"Error detecting hammer pattern: {e}")
            await asyncio.sleep(20)
            return False

    async def save_order_details(self, symbol, details):
        try:
            order_file = f"{symbol}_order.json"
            with open(order_file, 'w') as f:
                json.dump(details, f)
            logging.info(f"Order details saved for {symbol}")
        except Exception as e:
            logging.error(f"Error saving order details for {symbol}: {e}")
            await asyncio.sleep(20)

    async def order_exists(self, symbol):
        order_file = f"{symbol}_order.json"
        return os.path.exists(order_file)

    def remove_order_file(self, symbol):
        order_file = f"{symbol}_order.json"
        try:
            if os.path.exists(order_file):
                os.remove(order_file)
                logging.info(f"Order file removed for {symbol}")
        except Exception as e:
            logging.error(f"Error removing order file for {symbol}: {e}")

        finally: 
            return order_file
        

    # async def check_price_direction_change(self, symbol):
    #     try:
    #         # Fetch historical klines data
    #         result = await self.fetching_klines.fetch_historical_klines(
    #             symbol, start_time='2024-08-01', end_time='2070-01-01'
    #         )
            
    #         # Unpack the result, assuming it might be a tuple or other structure
    #         tema_9days, ma_7days, ma_14days, ma_30days, current_price = result
            
    #         # Log the result types and values for debugging
    #         logging.info(f"Fetched data for {symbol}: tema_9days={tema_9days}, ma_7days={ma_7days}, ma_14days={ma_14days}, ma_30days={ma_30days}, current_price={current_price}")
            
    #         # Ensure values are numerical and not hashable types
    #         def is_numerical(val):
    #             return isinstance(val, (int, float))
            
    #         if not all(is_numerical(val) for val in [tema_9days, ma_7days, ma_14days, ma_30days, current_price]):
    #             logging.warning(f"Non-numerical data for {symbol}. Cannot determine price direction.")
    #             return False
            
    #         # Perform comparisons
    #         return tema_9days != ma_7days and tema_9days != ma_14days and tema_9days != ma_30days
        
    #     except Exception as e:
    #         logging.error(f"Error checking price direction change for {symbol}: {e}")
    #         await asyncio.sleep(20)
    #         return False

    async def validate_data(self, symbol: str, data: pd.DataFrame, retries: int = 3) -> Tuple[bool, str]:
        # If you still want to check if `data` is a DataFrame, do it here
        if not isinstance(data, pd.DataFrame):
            logging.error(f"[{symbol}] Validation failed: Data is not a DataFrame")
            await asyncio.sleep(20)
            return False, "Data is not a DataFrame"

        attempt = 0

        while attempt < retries:
            attempt += 1

            if data.empty:
                logging.warning(f"[{symbol}] DataFrame is empty after cleaning. Retry attempt {attempt}/{retries}.")
                await asyncio.sleep(20)
                continue

            required_columns = ['open', 'high', 'low', 'close', 'volume']

            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                missing_columns_str = ', '.join(missing_columns)
                logging.error(f"[{symbol}] Validation failed: Data does not contain required columns: {missing_columns_str}")
                await asyncio.sleep(20)
                return False, f"Data does not contain required columns: {missing_columns_str}"

            nan_columns = data[required_columns].isna().any()
            if nan_columns.any():
                nan_column_names = [col for col, has_nan in nan_columns.items() if has_nan]
                nan_column_names_str = ', '.join([str(col) for col in nan_column_names])
                logging.error(f"[{symbol}] Validation failed: Data contains NaN values in columns: {nan_column_names_str}")
                await asyncio.sleep(20)
                return False, f"Data contains NaN values in columns: {nan_column_names_str}"

            non_numeric_columns = [col for col in required_columns if not pd.api.types.is_numeric_dtype(data[col])]
            if non_numeric_columns:
                non_numeric_columns_str = ', '.join(non_numeric_columns)
                logging.error(f"[{symbol}] Validation failed: Columns must be numeric: {non_numeric_columns_str}")
                await asyncio.sleep(20)
                return False, f"Columns must be numeric: {non_numeric_columns_str}"

            if not all(data['open'] >= data['low']) or not all(data['high'] >= data['close']) or not all(data['low'] <= data['high']):
                logging.error(f"[{symbol}] Validation failed: OHLCV data is logically inconsistent")
                await asyncio.sleep(20)
                return False, "OHLCV data is logically inconsistent"

            if any(data['volume'] <= 0):
                logging.error(f"[{symbol}] Validation failed: Volume contains negative values")
                await asyncio.sleep(20)
                return False, "Volume contains negative values"

            if not data.index.is_monotonic_increasing:
                logging.warning(f"[{symbol}] Data warning: Timestamp index is not sorted. This might be an issue with data integrity.")
                return False, "Timestamp index is not sorted"

            logging.info(f"[{symbol}] Data validated successfully on attempt {attempt}/{retries}.")
            return True, "Data validated successfully"

        logging.error(f"[{symbol}] Data validation failed after {retries} attempts.")
        await asyncio.sleep(20)
        return False, "Data validation failed after retries"

    async def retry_with_delay(self, func, *args, delay=20, max_retries=5, **kwargs):
        attempt = 0
        while attempt < max_retries:
            attempt += 1
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                logging.error(f"Attempt {attempt}/{max_retries} failed: {e}")
                await asyncio.sleep(20)
                if attempt < max_retries:
                    await asyncio.sleep(delay)
        raise Exception(f"Failed after {max_retries} attempts")
    




    async def close(self):
            """Close the Binance client session."""
            if self.client:
                await self.client.close_connection()
                logging.info("Binance client connection closed.")