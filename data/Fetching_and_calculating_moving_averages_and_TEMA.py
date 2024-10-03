import logging
import pandas as pd
from .Fetching_historical_OHCL_Values import KlineDataProcessor
import asyncio
import datetime
# import talib
import os
import dotenv
from binance.client import AsyncClient
from Order_handlers.OrderPlacement import OrderPlacement
# from binance.client import Client as BinanceClient
from datetime import datetime, timedelta
# from main import ProcessSymbol
from dotenv import load_dotenv  # Load environmental variables from .env file if present
from typing import List, Dict, Optional, Union, Sequence
from utils import TradingUtils
from colorama import Fore, Back, Style, init
from logger_config import get_logger


logging.basicConfig(level=logging.INFO)

# Get the logger for this module
logger = get_logger(__name__, blue_background=True)

dotenv.load_dotenv()  # Load environment variables from a .env file
# Load API credentials from environment variables
api_key = os.getenv('BINANCE_API_KEY')
api_secret = os.getenv('BINANCE_API_SECRET')

client = AsyncClient(api_key=api_key, api_secret=api_secret)

# Assuming 'client' is an instance of AsyncClient
print(dir(client))


class MovingAverageTrader:
    def __init__(self, client, symbol, window_9=9, window_30=30, symbols_file_path="T:\\Trading BOT Crypto - Profit\\symbols_orders_initializations.txt", klines_processor_configuration=None):
        self.client = client
        self.symbol = symbol
        self.window_9 = window_9
        self.window_30 = window_30
        self.klines = None
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.ma_values = {}
        self.symbols_file_path = symbols_file_path

        # Get API key and secret
        self.api_key = os.getenv('BINANCE_API_KEY')
        self.api_secret = os.getenv('BINANCE_API_SECRET')

        if not self.api_key or not self.api_secret:
            raise ValueError("API key and secret must be provided")

        self.validated_data = TradingUtils(self.client, symbol)
        self.order_placement = OrderPlacement(self.client, symbol)
        self.klines_processor_configuration = klines_processor_configuration

        # Initialize KlineDataProcessor with correct lambda
        self.kline_data_processor = KlineDataProcessor(
            client=client,
            moving_average_trader_class=lambda symbol, df, symbols_file_path=self.symbols_file_path: self.calculate_moving_averages(
                symbol, df, ma_windows={'ma_9': self.window_9, 'ma_30': self.window_30}, symbols_file_path=symbols_file_path
            ),
            symbols_file_path=self.symbols_file_path,
            api_key=self.api_key,
            api_secret=self.api_secret,
            max_requests=1000,
            start_time="2024-08-29",
            end_time="2100-01-01",
            fetch_attempts=3,
            cool_off=60,
            stop_if_fetched=False,
            log_retries=True,
            signal_trigger=None
        )

        if not self.api_key or not self.api_secret:
            self.logger.error("API key and secret must be provided")
            raise ValueError("API key and secret must be provided")
        else:
            self.logger.info("API key and secret loaded successfully")

        # Check client initialization
        # await self.check_client_initialized()

    async def check_client_initialized(self):
        """Check if the client is initialized properly. If not, create an async client."""
        if self.client is None:
            self.logger.info("Client is not initialized. Attempting to create an async client...")
            self.client = await self.initialize_async_client(client)

            if self.client is None:
                raise ValueError("Failed to create an async client. Please provide valid credentials.")
        else:
            self.logger.info("Client is already initialized.")

    async def initialize_async_client(self, client):
        """Initialize the Binance async client."""
        
        try:
            if client is None:
                return await AsyncClient.create(self.api_key, self.api_secret)
        except Exception as e:
            self.logger.error(f"Error initializing async client: {e} in MovingAverageTrader")
            return None

    async def fetch_klines(self, interval='1m', limit=200):
        """Fetch klines data and assign it to self.klines."""
        await self.check_client_initialized()  # Ensure the client is initialized

        if self.client is None:
            self.logger.error("Client is not initialized. Cannot fetch klines.")
            return None

        self.klines = await self.client.get_historical_klines(self.symbol, interval, limit=limit)
        return self.klines

    def process_klines(self):
        """Process the fetched klines to create a DataFrame and assign last_candles."""
        if self.klines is not None:
            # Assuming self.klines is a list of lists as fetched from the API
            df = pd.DataFrame(self.klines, columns=['open time', 'open', 'High', 'Low', 'close', 'Volume', 'close time', 'Quote asset volume', 'Number of trades', 'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'])
            df['close'] = df['close'].astype(float)
            df['open'] = df['open'].astype(float)
            last_candles = df[['open', 'close']].tail(self.window_30)  # For example, using the last 30 candles
            return last_candles
        else:
            self.logger.warning("Klines data is not available. Please fetch klines first.")
            return None

    async def fetch_current_close(self, last_candles: pd.DataFrame, index: int) -> float:
        """Fetches the current close price for a given index."""
        return last_candles['close'].iloc[-index]

    async def fetch_previous_open(self, last_candles: pd.DataFrame, index: int) -> float:
        """Fetches the previous open price for a given index."""
        return last_candles['open'].iloc[-index - 1]
    
    async def calculate_tema_and_ma_async(self, symbols, df: Optional[Dict[str, pd.DataFrame]] = None, 
                                      window_tema: Optional[Sequence[int]] = None, 
                                      ma_windows: Optional[Sequence[int]] = None, 
                                      confirmation_intervals: Optional[List[int]] = None, 
                                      confirmation_checks: Optional[List[int]] = None):

        if not symbols:
            self.logger.error("Symbols list is empty.")
            return {}

        symbol = symbols[0]  # Single symbol

        if df is None:
            df = {}

        # Initialize empty DataFrame if symbol not in df
        if symbol not in df:
            df[symbol] = pd.DataFrame()

        # Fetch klines if necessary
        if df[symbol].empty or 'close' not in df[symbol].columns:
            try:
                klines = await self.kline_data_processor.process_data()
                if klines is None or not isinstance(klines, tuple) or len(klines) != 2:
                    self.logger.error(f"Invalid klines data for {symbol}.")
                    return {}

                _, klines_df = klines
                if klines_df.empty:
                    self.logger.error(f"Empty DataFrame returned for {symbol}.")
                    return {}

                df[symbol] = klines_df
            except Exception as e:
                self.logger.error(f"Error fetching or processing data for {symbol}: {str(e)}")
                return {}

        # Ensure the data is valid
        if df[symbol].empty or 'close' not in df[symbol].columns:
            self.logger.error(f"DataFrame is invalid for {symbol}. Missing 'close' column or empty data.")
            return {}

        # Set default values for parameters and ensure they are lists
        window_tema = list(window_tema or [9, 20])
        ma_windows = list(ma_windows or [7, 14, 30])
        confirmation_intervals = list(confirmation_intervals or [10, 3, 15])
        confirmation_checks = list(confirmation_checks or [5, 20, 1])

        self.logger.info(f"Using TEMA windows: {window_tema}")
        self.logger.info(f"Using MA windows: {ma_windows}")

        results = {}

        try:
            # Calculate TEMA
            tema_results = {}
            for w in window_tema:
                # Assuming you have a method calculate_tema
                if not await self.calculate_tema(df[symbol], [w]):
                    self.logger.warning(f"Failed to calculate TEMA for window {w} on {symbol}, continuing.")
                    continue
                tema_results[f'TEMA_{w}'] = df[symbol][f'TEMA_{w}'].iloc[-1]

            results.update(tema_results)

            # Calculate moving averages
            ma_values = await self.calculate_moving_averages(symbol, df[symbol], ma_windows=ma_windows, symbols_file_path=self.symbols_file_path)
            if not isinstance(ma_values, dict) or not ma_values:
                self.logger.error(f"Invalid or empty moving averages data for {symbol}.")
                return {}

            results.update(ma_values)

            self.logger.info(f"Calculated results for {symbol}: {results}")

            # Continue symbol processing
            await self.process_symbol(symbols, df[symbol], window_tema, ma_windows)  # Now both are lists

            return results

        except Exception as e:
            self.logger.error(f"Error while calculating for {symbol}: {str(e)}", exc_info=True)
            return {}

    async def clean_dataframe(self, symbols: List[str], df: Optional[pd.DataFrame], klines: Optional[List]) -> pd.DataFrame:
        # Check if df is None and if klines is provided
        if df is None and klines:
            try:
                # Fetch DataFrame(s) using klines_processor
                df_or_dict = await self.kline_data_processor.fetch_live_klines()
                
                # If the result is a dictionary, merge DataFrames for each symbol
                if isinstance(df_or_dict, dict):
                    df = pd.concat([df_or_dict.get(symbol, pd.DataFrame()) for symbol in symbols], ignore_index=True)
                else:
                    df = df_or_dict

            except Exception as e:
                self.logger.error(f"Error fetching data to convert to DataFrame: {str(e)}")
                await asyncio.sleep(20)
                return pd.DataFrame()

        # Check if df is None or empty after attempting to fetch
        if df is None or df.empty:
            self.logger.error("DataFrame is None or empty after calling klines_to_dataframe. Returning empty DataFrame.")
            await asyncio.sleep(20)
            return pd.DataFrame()

        try:
            # Clean the DataFrame
            self.logger.info(f"Starting to clean DataFrame. Initial shape: {df.shape}")

            # Convert all columns to numeric values, coercing errors to NaN
            df = df.apply(pd.to_numeric, errors='coerce')

            # Drop rows where 'close' column is NaN
            df.dropna(subset=['close'], inplace=True)

            # Check if DataFrame is empty after cleaning
            if df.empty:
                self.logger.warning("DataFrame is empty after cleaning. Returning empty DataFrame.")
                await asyncio.sleep(20)
                return pd.DataFrame()

            # Log the shape of the cleaned DataFrame
            self.logger.info(f"DataFrame cleaned successfully. Final shape: {df.shape}")

            return df

        except Exception as e:
            self.logger.error(f"Unexpected error while cleaning DataFrame: {str(e)}")
            await asyncio.sleep(20)
            return pd.DataFrame()

    async def retry_delay_on_error(self, delay= 3, return_type: str = "default") -> Union[Dict[str, float], bool]:
        """
        Utility function for handling retries and adding delays in case of errors.
        It can return different types based on the `return_type` argument.
        
        Parameters:
        - delay: The delay in seconds to wait before retrying.
        - return_type: The type of value to return. "dict" returns an empty dictionary, otherwise returns False.
        
        Returns:
        - Either an empty dictionary or False based on the `return_type`.
        """
        await asyncio.sleep(delay)
        
        if return_type == "dict":
            return {}  # Return empty dictionary for cases expecting a dict
        return False  # Default behavior

    async def calculate_tema(self, df: pd.DataFrame, window_temas: List[int]) -> bool:
        try:
            if df.empty or 'close' not in df.columns:
                logger.warning("DataFrame invalid. Cannot calculate TEMA.")
                return False

            required_rows = max(window_temas) * 3
            if len(df) < required_rows:
                logger.warning("Not enough data for TEMA calculation.")
                return False

            for window in window_temas:
                # Ensure that the current window is valid
                if window > len(df):
                    logger.warning(f"Window size {window} exceeds available data length for TEMA calculation.")
                    continue
                
                # Calculate EMA values
                df[f'ema_{window}'] = df['close'].ewm(span=window, adjust=False).mean()
                df[f'ema_{window}_ema'] = df[f'ema_{window}'].ewm(span=window, adjust=False).mean()
                df[f'ema_{window}_ema_ema'] = df[f'ema_{window}_ema'].ewm(span=window, adjust=False).mean()
                df[f'TEMA_{window}'] = 3 * (df[f'ema_{window}'] - df[f'ema_{window}_ema']) + df[f'ema_{window}_ema_ema']

                # Log the calculated TEMA value for this window
                logger.info(f"Calculated TEMA_{window} for last row: {df[f'TEMA_{window}'].iloc[-1]}")

            return True
        except Exception as e:
            logger.error(f"Error in calculate_tema: {str(e)}")
            await self.retry_delay_on_error(20, return_type="bool")
            return False

    async def validate_data(self, df: pd.DataFrame) -> bool:
        try:
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                logger.warning(f"DataFrame is missing required columns: {', '.join(missing_cols)}.")
                return False

            logger.info(f"DataFrame columns present: {', '.join(df.columns)}")
            return True
        except Exception as e:
            logger.error(f"Error validating data: {str(e)}")
            await asyncio.sleep(20)
            return False

    async def calculate_moving_averages(self, symbol, df, ma_windows, symbols_file_path) -> Dict[str, float]:
        if not self.api_key or not self.api_secret:
            self.logger.error("API key and secret must be provided as non-None strings.")
            return {}

        ma_values = {}
        try:
            if not isinstance(df, pd.DataFrame) or df.empty:
                self.logger.error("Provided DataFrame is not valid or is empty.")
                return {}

            for window in ma_windows:
                col_name = f'MA_{window}'
                if len(df) >= window:
                    df[col_name] = df['close'].rolling(window=window).mean()
                    ma_values[col_name] = df[col_name].iloc[-1]
                else:
                    ma_values[col_name] = float('nan')
                    self.logger.warning(f"Not enough data for {symbol} to calculate MA_{window}")

            return ma_values

        except Exception as e:
            self.logger.error(f"Error calculating moving averages for {symbol}: {str(e)}")
            return {}

    async def check_first_condition(self, symbol: List[str], df: pd.DataFrame) -> bool:
        try:
            tema_20_value = df['TEMA_20'].iloc[-1]
            tema_9_value = df['TEMA_9'].iloc[-1]

            if tema_20_value > tema_9_value:
                self.logger.info(f"TEMA 20 > TEMA 9 for {symbol}. Condition met.")
                return True
            else:
                self.logger.info(f"TEMA 20 <= TEMA 9 for {symbol}. Checking candlestick prices...")
                # Ensure you define a check_duration, e.g., 60 for 60 seconds
                candlestick_valid = await self.check_candlestick_prices(symbol)
                return candlestick_valid
        except Exception as e:
            self.logger.error(f"Error in check_first_condition for {symbol}: {str(e)}")
            return False

    async def confirm_first_condition(self, symbol: str, df: pd.DataFrame) -> bool:
        retry_attempts = 10  # Adjusted number of retry attempts
        initial_retry_delay = 1  # Initial retry delay (in seconds)
        max_retry_delay = 30  # Max retry delay to prevent overloading the API
        retry_delay = initial_retry_delay

        for attempt in range(1, retry_attempts + 1):
            try:
                self.logger.debug(f"Attempt {attempt}: Received DataFrame for {symbol} of type {type(df)}")

                # Check for None
                if df is None:
                    self.logger.error(f"Attempt {attempt}: Received None for {symbol}. Retrying in {retry_delay} seconds...")
                    await asyncio.sleep(retry_delay)
                    continue

                # Check if df is a DataFrame
                if not isinstance(df, pd.DataFrame):
                    self.logger.error(f"Attempt {attempt}: Expected DataFrame but received {type(df)} for {symbol}. Retrying in {retry_delay} seconds...")
                    await asyncio.sleep(retry_delay)
                    continue

                # Check if DataFrame is empty
                if df.empty:
                    self.logger.info(f"Attempt {attempt}: DataFrame is empty for {symbol}. Retrying in {retry_delay} seconds...")
                    await asyncio.sleep(retry_delay)
                    continue

                # Validate required columns
                required_columns = ['close', 'open', 'high', 'low', 'TEMA_9', 'TEMA_20']  # Make sure all necessary columns are present
                missing_columns = [col for col in required_columns if col not in df.columns]
                if missing_columns:
                    self.logger.error(f"Attempt {attempt}: DataFrame for {symbol} is missing required columns: {missing_columns}. Retrying in {retry_delay} seconds...")
                    await asyncio.sleep(retry_delay)
                    continue

                # Log details of the DataFrame
                self.logger.debug(f"Attempt {attempt}: DataFrame head for {symbol}:\n{df.head()}")

                # Check the first condition using the helper function
                condition_met = await self.check_first_condition_task(symbol, df)
                if condition_met:
                    self.logger.info(f"Attempt {attempt}: First short condition confirmed for {symbol}.")
                    return True

                self.logger.info(f"Attempt {attempt}: First condition not met for {symbol}. Retrying in {retry_delay} seconds...")
                await asyncio.sleep(retry_delay)

                # Increment retry delay (with a maximum cap)
                retry_delay = min(retry_delay * 2, max_retry_delay)

            except KeyError as ke:
                self.logger.error(f"Attempt {attempt}: KeyError while accessing DataFrame for {symbol}: {str(ke)}. Retrying in {retry_delay} seconds...")
                await asyncio.sleep(retry_delay)
            except Exception as e:
                self.logger.error(f"Attempt {attempt}: Unexpected error in confirm_first_condition for {symbol}: {str(e)}. Retrying in {retry_delay} seconds...")
                await asyncio.sleep(retry_delay)

        self.logger.warning(f"First short condition not confirmed for {symbol} after {retry_attempts} attempts.")
        return False

    async def check_second_condition(self, symbol: str, df: pd.DataFrame) -> bool:
        try:
            # Validate that the DataFrame has the necessary columns and is not empty
            if df is not None and 'MA_7' in df.columns and 'MA_14' in df.columns and 'MA_30' in df.columns and not df.empty:
                
                # Extract the latest moving averages from the DataFrame
                ma_7_value = df['MA_7'].iloc[-1]
                ma_14_value = df['MA_14'].iloc[-1]
                ma_30_value = df['MA_30'].iloc[-1]

                # Log the current MA values for debugging
                logger.debug(f"Checking second condition for {symbol} with MA values: MA_30={ma_30_value}, MA_14={ma_14_value}, MA_7={ma_7_value}")

                # Validate that the values are not NaN
                if pd.isna(ma_30_value) or pd.isna(ma_14_value) or pd.isna(ma_7_value):
                    logger.error(f"Invalid MA values detected for {symbol}: MA_30={ma_30_value}, MA_14={ma_14_value}, MA_7={ma_7_value}. Retrying...")
                    result = await self.retry_delay_on_error(20, return_type="bool")  # Retry after delay
                    return isinstance(result, bool) and result

                # Check the condition MA_30 < MA_14 < MA_7
                condition_result = (ma_30_value < ma_14_value) and (ma_14_value < ma_7_value)
                logger.info(f"Second condition result for {symbol} (MA_30 < MA_14 < MA_7): {condition_result}")

                return condition_result
            else:
                logger.error(f"DataFrame missing required MA columns or is empty for {symbol}.")
                return False

        except Exception as e:
            logger.error(f"Error in check_second_condition for {symbol}: {str(e)}")
            result = await self.retry_delay_on_error(20, return_type="bool")
            return isinstance(result, bool) and result

    async def check_second_condition_task(self, symbol: str, df: pd.DataFrame, duration_minutes: int = 5) -> bool:
        # Cap duration between 2 and 10 minutes
        duration_minutes = max(2, min(duration_minutes, 10))

        # Get the current local time
        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=duration_minutes)  # Use timedelta here

        logger.info(f"Starting check for {symbol} at {start_time.strftime('%Y-%m-%d %H:%M:%S')} for {duration_minutes} minutes.")

        while datetime.now() < end_time:
            try:
                if isinstance(df, pd.DataFrame) and not df.empty:
                    logger.debug(f"Checking second condition for {symbol} with DataFrame")
                    condition_result = await self.check_second_condition(symbol, df)
                    if condition_result:
                        return True
                    else:
                        logger.info(f"Second condition not met for {symbol}, continuing checks...")
                else:
                    logger.warning(f"Invalid or empty DataFrame provided for {symbol}.")
                    return False

                await asyncio.sleep(10)

            except Exception as e:
                logger.error(f"Error checking second condition for {symbol}: {str(e)}")
                return False
        
        logger.info(f"Timed out checking second condition for {symbol} after {duration_minutes} minutes.")
        return False

    async def check_candlestick_prices(self, symbol, check_duration: int = 5, max_attempts: int = 7) -> bool:
        attempt_count = 0

        while attempt_count < max_attempts:
            try:
                start_time = datetime.now()
                consecutive_candles = 0
                confirmation_needed = 4  # For stronger confirmation
                min_candles_for_opportunity = 5  # Minimum for initial opportunity check

                while (datetime.now() - start_time).total_seconds() < check_duration * 60:
                    # Fetch the latest data to update the DataFrame
                    df = await self.fetch_klines()

                    if isinstance(df, list):
                        df = pd.DataFrame(df)  # Convert to DataFrame if it's a list

                    if not isinstance(df, pd.DataFrame) or df.empty or len(df) < 8:
                        logger.info(f"Not enough candles for analysis for {symbol}. Minimum of 8 required.")
                        await asyncio.sleep(1)
                        continue

                    last_candles = df.iloc[-8:].copy()  # Get the last 8 candles

                    for i in range(1, 8):
                        if i >= len(last_candles):
                            logger.warning(f"Skipping invalid index for {symbol}: Not enough candle data for iteration {i}.")
                            break  # Exit for loop if there's not enough data

                        # Fetch current close and previous open asynchronously
                        current_close = await self.fetch_current_close(last_candles, i)
                        previous_open = await self.fetch_previous_open(last_candles, i)

                        # Check condition for short opportunity
                        if current_close < previous_open:
                            consecutive_candles += 1
                            logger.info(f"Candlestick check: Current close ({current_close}) < Previous open ({previous_open}). Consecutive checks: {consecutive_candles} for {symbol}.")
                        else:
                            consecutive_candles = 0
                            logger.info(f"Candlestick check for {symbol}: Condition not met, resetting checks.")

                        if consecutive_candles >= min_candles_for_opportunity:
                            logger.info(f"Partial Short Condition confirmed for {symbol} with {consecutive_candles} consecutive candles. Verifying additional conditions for confirmation...")

                            confirmed = False
                            for _ in range(confirmation_needed):
                                await asyncio.sleep(1)
                                df = await self.kline_data_processor.fetch_live_klines()  # Fetch again to confirm with the latest data

                                if isinstance(df, list):
                                    df = pd.DataFrame(df)  # Convert to DataFrame if it's a list

                                if not isinstance(df, pd.DataFrame) or df.empty or len(df) < 7 + confirmation_needed:
                                    logger.info(f"Not enough data to confirm signal for {symbol}. Waiting for additional candles.")
                                    continue

                                confirmed = True
                                logger.info(f"Candlestick check: Condition met and confirmed for {symbol}.")
                                break  # Break if confirmed

                            if confirmed:
                                logger.info(f"Position size may be increased based on further confirmations for {symbol}.")
                                return True  # Return true if condition is confirmed

                        # Log different strength levels of opportunity
                        if consecutive_candles >= 6:
                            logger.info(f"Impressive short opportunity for {symbol}: {consecutive_candles} consecutive candles.")
                        elif consecutive_candles >= 5:
                            logger.info(f"Strong short opportunity for {symbol}: {consecutive_candles} consecutive candles.")

                    await asyncio.sleep(2)

                logger.info(f"Candlestick check: Time limit exceeded for {symbol}. Moving to the next attempt.")
                attempt_count += 1
                logger.info(f"Retry attempt {attempt_count}/{max_attempts}")

            except ValueError as ve:
                logger.error(f"ValueError while checking candlestick prices for {symbol}: {str(ve)}")
                await asyncio.sleep(20)
            except Exception as e:
                logger.error(f"Unexpected error while checking candlestick prices for {symbol}: {str(e)}. Data type: {type(df)}")
                await asyncio.sleep(20)

        logger.info(f"Maximum attempts reached for {symbol}. Reverting to process_symbol.")
        try:
            if not isinstance(df, pd.DataFrame):
                logger.error(f"Invalid DataFrame for processing symbol {symbol}.")
                return False
            window_tema = [9, 20]  # Example values for TEMA
            ma_windows = [7, 14, 30]  # Example values for moving averages
            return await self.process_symbol(symbol, df=df, klines=self.klines, window_tema=window_tema, ma_windows=ma_windows)
        except Exception as e:
            logger.error(f"Error while processing symbol after max attempts: {str(e)}")
            await asyncio.sleep(2)
            return False

    async def process_symbol(self, symbols: List[str], df: pd.DataFrame, window_tema: List[int], ma_windows: List[int], klines: Optional[List] = None) -> bool:
        try:
            logger.info(f"Processing symbols: {symbols}")

            # Clean and validate DataFrame
            df = await self.clean_dataframe(symbols=symbols, df=df, klines=klines)
            if df is None or df.empty:
                logger.error(f"DataFrame validation failed for {symbols}.")
                await asyncio.sleep(20)
                return False

            if not await self.validate_data(df):
                logger.error(f"DataFrame validation failed for {symbols}.")
                await asyncio.sleep(20)
                return False

            # Calculate TEMA
            if not await self.calculate_tema(df, window_tema):
                logger.error(f"Failed to calculate TEMA for {symbols}.")
                await asyncio.sleep(20)
                return False

            # Calculate Moving Averages
            ma_values = await self.calculate_moving_averages(symbols[0], df, ma_windows, symbols_file_path=self.symbols_file_path)
            if not isinstance(ma_values, dict) or not ma_values:
                logger.error(f"Failed to calculate moving averages for {symbols}.")
                await asyncio.sleep(20)
                return False

            # Check all conditions concurrently
            tasks = [
                self.check_first_condition_task(symbols[0], df),
                self.check_second_condition_task(symbols[0], df),
                self.check_third_condition_task(symbols[0], df)
            ]
            condition_results = await asyncio.gather(*tasks)

            # Proceed only if all conditions are met
            if all(condition_results):
                logger.info(f"All conditions met for {symbols}. Short signal confirmed.")
                return True

            logger.info(f"Conditions not met for {symbols}.")
            return False

        except (asyncio.TimeoutError, KeyError, ValueError, TypeError) as e:
            logger.error(f"Error while processing {symbols}: {str(e)}")
            await asyncio.sleep(20)
            return False
        except Exception as e:
            logger.error(f"Unexpected error while processing {symbols}: {str(e)}")
            await asyncio.sleep(20)
            return False

    async def check_first_condition_task(self, symbol: str, df: pd.DataFrame) -> bool:
        try:
            # Validate that the DataFrame has the necessary columns and is not empty
            if df is not None and 'TEMA_20' in df.columns and 'TEMA_9' in df.columns and not df.empty:
                tema_20_value = df['TEMA_20'].iloc[-1]
                tema_9_value = df['TEMA_9'].iloc[-1]
                logger.debug(f"Checking first condition for {symbol} with TEMA values: TEMA_20={tema_20_value}, TEMA_9={tema_9_value}")

                # Perform candle analysis for at least 3 consecutive candles
                consecutive_candle_count = 3
                last_candles = df.tail(consecutive_candle_count + 1)  # +1 to compare closure with previous

                # Ensure we have enough data
                if len(last_candles) < consecutive_candle_count + 1:
                    logger.warning(f"Not enough candles to perform consecutive analysis for {symbol}.")
                    return False

                # Iterate over the last 3 candles and calculate the differences
                for i in range(-consecutive_candle_count, 0):
                    open_value = last_candles['open'].iloc[i]
                    high_value = last_candles['high'].iloc[i]
                    low_value = last_candles['low'].iloc[i]
                    close_value = last_candles['close'].iloc[i]
                    prev_close_value = last_candles['close'].iloc[i - 1]

                    # Calculate the difference between high/open and open/low
                    high_open_diff = high_value - open_value
                    open_low_diff = open_value - low_value

                    # Adjust your condition check here, using specific thresholds
                    # Example: Check if high_open_diff > 0.01 * open_value (1% of the open value)
                    if high_open_diff <= 0.01 * open_value:
                        logger.info(f"High-Open diff not significant for {symbol} on candle index {i}.")
                        return False

                    # Example: Check if open_low_diff > 0.01 * open_value (1% of the open value)
                    if open_low_diff <= 0.01 * open_value:
                        logger.info(f"Open-Low diff not significant for {symbol} on candle index {i}.")
                        return False

                    # Check if the current candle closure is lower than the previous candle closure
                    if close_value >= prev_close_value:
                        logger.info(f"Closure values not decreasing for {symbol} on candle index {i}.")
                        return False

                    logger.debug(f"Candle {i} for {symbol}: High-Open diff={high_open_diff}, Open-Low diff={open_low_diff}, Close={close_value}, Prev Close={prev_close_value}")

                # Call confirm_first_condition directly with TEMA values
                return await self.confirm_first_condition(symbol, tema_9_value)  # Adjust this as per your logic
            else:
                logger.warning(f"TEMA values not available or DataFrame is empty for {symbol}.")
                return False

        except Exception as e:
            logger.error(f"Error checking first condition for {symbol}: {str(e)}")
            return False

    async def confirm_second_condition(self, symbol: str, df: pd.DataFrame) -> bool:
        retry_attempts = 10  # Reduced number of attempts for efficiency
        base_delay = 0.5  # Base delay in seconds

        for attempt in range(retry_attempts):
            try:
                logger.info(f"Attempting to check second condition for {symbol} (Attempt {attempt + 1})...")
                
                # Pass the DataFrame (df) directly to check_second_condition
                condition_result = await self.check_second_condition(symbol, df)
                
                if condition_result:
                    logger.info(f"Second condition confirmed on attempt {attempt + 1} for {symbol}.")
                    return True
                else:
                    logger.info(f"Second condition check failed for {symbol}. Retrying...")

                await asyncio.sleep(base_delay * (2 ** attempt))  # Exponential backoff
            except Exception as e:
                logger.error(f"Error during second condition check for {symbol}: {str(e)}")
                await asyncio.sleep(20)  # Brief sleep to avoid tight loop in case of consistent errors

        logger.warning(f"Second condition could not be confirmed for {symbol} after {retry_attempts} attempts.")
        return False

    async def check_third_condition_task(self, symbol: str, df: pd.DataFrame) -> bool:
        try:
            if df is not None and not df.empty:
                logger.debug(f"Checking third condition for {symbol} with DataFrame of shape {df.shape}")
                check_duration = 60  # Specify check_duration
                return await self.check_candlestick_prices(symbol, check_duration) 
            else:
                logger.warning(f"DataFrame is empty or not valid for {symbol}.")
                return False
        except Exception as e:
            logger.error(f"Error checking third condition for {symbol}: {str(e)}")
            return False

    async def orchestrate_short_conditions(self, symbol: str, df: pd.DataFrame, ma_values: List[float]) -> bool:
        try:
            # Validate inputs
            if not isinstance(symbol, str) or not symbol:
                logger.error("Invalid symbol provided.")
                return False
            
            if df is None or df.empty:
                logger.error(f"DataFrame is empty or None for {symbol}.")
                return False

            if not isinstance(ma_values, dict) or not ma_values:
                logger.error(f"Invalid moving averages provided for {symbol}: {ma_values}")
                return False

            # Wrap symbol in a list for task calls
            symbol_list = [symbol]

            # Run all condition tasks concurrently
            first_condition_task = asyncio.create_task(self.check_first_condition_task(symbol_list[0], df))
            second_condition_task = asyncio.create_task(self.check_second_condition_task(symbol_list[0], df))
            third_condition_task = asyncio.create_task(self.check_third_condition_task(symbol_list[0], df))

            # Wait for all tasks to complete
            first_condition_met, second_condition_met, third_condition_met = await asyncio.gather(
                first_condition_task, second_condition_task, third_condition_task
            )

            # Check the results of the conditions
            if first_condition_met and second_condition_met and third_condition_met:
                logger.info(f"All conditions met for {symbol}. Short signal confirmed.")
                return True

            await asyncio.sleep(2)
            logger.info(f"Conditions not met for {symbol}.")
            return False

        except asyncio.CancelledError:
            logger.warning(f"Short condition orchestration was cancelled for {symbol}.")
            return False
        except Exception as e:
            logger.error(f"Unexpected error in orchestrate_short_conditions for {symbol}: {str(e)}")
            return False

    async def close_client(self) -> bool:
        """close the Binance client session."""
        try:
            if self.client:
                # Optionally check for pending tasks before closing
                # You might want to implement a way to check for ongoing operations here

                await self.client.close_connection()
                self.logger.info("Binance client connection closed successfully.")
                return True  # Indicate successful closure
            else:
                self.logger.warning("No client connection to close.")
                return False  # Indicate that no connection was available
        except (ConnectionError, asyncio.TimeoutError) as e:
            self.logger.error(f"Connection error while closing Binance client: {e}")
            return False  # Indicate failure due to connection error
        except Exception as e:
            self.logger.error(f"Unexpected error occurred while closing Binance client connection: {e}")
            return False  # Indicate failure for other unexpected errors
