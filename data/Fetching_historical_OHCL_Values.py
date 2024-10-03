import pandas as pd
import numpy as np
import logging
import os
import json
import websockets
import asyncio
from tabulate import tabulate
from colorama import Fore, Style
import traceback
from datetime import datetime, timedelta
import pytz
import time

import dotenv

from binance.exceptions import BinanceAPIException, BinanceRequestException
from binance.client import AsyncClient
from binance import BinanceSocketManager

# from utils import TradingUtils
from typing import List, Union, Optional, Callable, Dict, Literal, Tuple
from logger_config import get_logger, CustomLoggerAdapter



dotenv.load_dotenv()  # Load environment variables from a .env file

# Load API credentials from environment variables
api_key = os.getenv('BINANCE_API_KEY')
api_secret = os.getenv('BINANCE_API_SECRET')

# client = await AsyncClient.create(api_key, api_secret)


# Set the event loop policy to SelectorEventLoop for compatibility with aiodns
if os.name == 'nt':  # Check if the OS is Windows
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


# Configuration error check
if not api_key or not api_secret:
    raise EnvironmentError("Binance API credentials are not set in environment variables. Please check your .env file.")


# Get the logger for this module
logger = get_logger(__name__, blue_background=True)

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.DEBUG,  # Adjust the level as needed
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

#Instantiating the class TradingUtils for utility function to be ran on converted dataframe


class KlineDataProcessor:
    def __init__(self, client,
                 moving_average_trader_class: Optional[Callable] = None,
                 api_key: Optional[str] = None, api_secret: Optional[str] = None, 
                 symbols_file_path: str = "T:\\Trading BOT Crypto - Profit\\symbols_orders_initializations.txt", 
                 max_requests: int = 1000, start_time: str = "2024-08-29", end_time: str = "2100-01-01", 
                 fetch_attempts: int = 3, cool_off: Optional[float] = 5.0,  
                 stop_if_fetched: Optional[bool] = None, continuous_fetch: bool = False, 
                 log_retries: bool = True, fetch_delay: float = 10.0, 
                 signal_trigger: Optional[Callable] = None,
                 symbols: Optional[list] = None,
                 **kwargs):  
        
        self.client = client
        self.api_key = api_key
        self.api_secret = api_secret
        self.max_requests = max_requests
        self.start_time = start_time
        self.end_time = end_time
        self.fetch_attempts = fetch_attempts
        self.cool_off = cool_off if cool_off is not None else 5.0
        self.stop_if_fetched = stop_if_fetched
        self.log_retries = log_retries
        self.signal_trigger = signal_trigger
        self.continuous_fetch = continuous_fetch
        self.fetch_delay = fetch_delay
        self.symbols_file_path = symbols_file_path
        self.ma_values = {}
        self.tema_values = {}
        self.last_fetched_times = {}
        self.local_timezone = pytz.timezone('Europe/Bucharest')
        self.fetched_symbols = set()
        self.kline_buffer = []  # Buffer to store incoming kline data
        self.buffer_size = 350  # Required size for the DataFrame

        # Load symbols
        self.symbol = symbols if symbols else self.symbol_loader()
        self.colored_logger = CustomLoggerAdapter(logger)
        # Instantiate the KlineDataProcessor class
        
    async def _initialize_client(self, api_key=api_key, api_secret=api_secret, max_attempts=3):
        """
        Initialize the Binance AsyncClient and set it to the class instance.
        Retry initialization in case of failure.
        """
        attempt = 0
        while attempt < max_attempts:
            try:
                print(f"Initializing Binance client, attempt {attempt + 1}...")
                # Initialize the AsyncClient and assign it to the instance variable self.client
                self.client = await AsyncClient.create(api_key=api_key, api_secret=api_secret)
                print("Client initialized successfully.")
                return  # Exit once client is initialized
            except Exception as e:
                print(f"Error initializing Binance client on attempt {attempt + 1}: {e}")
                attempt += 1
                if attempt >= max_attempts:
                    print("Max attempts reached. Client initialization failed.")
                    return None
            await asyncio.sleep(2)  # Cool-off before retrying

    fetched_symbols = set()  # To track symbols that have finished fetching and processing

    def symbol_loader(self, reload_interval: int = 3600) -> Callable[[str], List[str]]:
        """
        A closure that returns a function to load symbols from a file, caching the result and 
        reloading only after the specified interval has passed.
        
        :param reload_interval: Time interval in seconds to reload symbols, default is 1 hour.
        :return: A function that takes a file path and returns a list of symbols.
        """
        _symbols_cache = None
        _last_load_time = 0  # Initialize to 0 so it reloads the first time

        def load_symbols(file_path: str) -> List[str]:
            nonlocal _symbols_cache, _last_load_time
            current_time = time.time()

            # Calculate time since last load and reload if the interval has passed
            if _symbols_cache is None or (current_time - _last_load_time) > reload_interval:
                logger.info(f"Reloading symbols after {reload_interval} seconds or on first load: {file_path}")
                try:
                    if os.path.exists(file_path):  # Check if it's a file path
                        with open(file_path, 'r') as file:
                            _symbols_cache = file.read().splitlines()  # Read symbols as lines
                        logger.info(f"Loaded symbols from file: {_symbols_cache}")
                    else:
                        logger.info(f"Treating input as a single symbol: {file_path}")
                        _symbols_cache = [file_path]

                    # Update last load time only when loading is successful
                    _last_load_time = current_time
                except Exception as e:
                    logger.error(f"Failed to read symbols from file or input: {e}")
                    return []
            
            return _symbols_cache

        return load_symbols


    async def initialize_moving_average_trader(self):
        try:
            from .Fetching_and_calculating_moving_averages_and_TEMA import MovingAverageTrader

            # Initialize Binance client
            self.client = await AsyncClient.create(api_key=self.api_key, api_secret=self.api_secret)

            # Initialize OrderPlacement instance with client and symbols
            from Order_handlers.OrderPlacement import OrderPlacement
            self.order_placement_instance = OrderPlacement(self.client, symbol=self.symbols_file_path)

            # Initialize MovingAverageTrader with necessary parameters
            self.moving_average_trader = MovingAverageTrader(
                self.client, 
                symbol=self.symbols_file_path, 
                window_9=9, 
                window_30=30
            )

            if not self.moving_average_trader:
                logger.error("Failed to initialize MovingAverageTrader.")
                return  # Exit early if initialization failed

            logger.info("MovingAverageTrader initialized successfully.")

            # Check ongoing positions for each unique symbol
            for symbol in self.moving_average_trader.symbol:
                ongoing_positions = await self.order_placement_instance.check_ongoing_positions(symbol)

                if ongoing_positions:
                    logger.info(f"Skipping {symbol} due to ongoing positions.")
                    continue  # Skip processing for this symbol if there are ongoing positions
                else:
                    logger.info(f"No ongoing positions for {symbol}. Proceeding with trading logic.")

        except Exception as e:
            logger.error(f"Error initializing MovingAverageTrader: {e}")
        finally:
            # Close client connection only when all processing is done
            if self.client:
                await self.client.close_connection()
                logger.info("Binance client connection closed.")


    

    async def fetch_klines(self):
        fetched_count = 0
        last_active_time = datetime.now(self.local_timezone).timestamp()
        dataframes = []

        # Check if self.symbol is a list of symbols
        if isinstance(self.symbol, list):
            logger.info(f"Loaded symbols: {self.symbol}")
        else:
            logger.error("Symbols should be a list.")
            return pd.DataFrame()  # Return an empty DataFrame

        # Check if the moving average trader is initialized
        if self.moving_average_trader is not None:
            await self.moving_average_trader.check_client_initialized()
        else:
            self.colored_logger.warning("Moving average trader not initialized. Client check skipped.")

        # Check if the client is initialized
        if self.client is None:
            logger.error("Client not initialized. Cannot fetch klines.")
            return pd.DataFrame()

        async def fetch_single_kline(symbol):
            nonlocal fetched_count, last_active_time

            if symbol in self.fetched_symbols:
                logger.info(f"Klines for {symbol} already fetched and processed, skipping.")
                return pd.DataFrame()  # Return an empty DataFrame

            for attempt in range(self.fetch_attempts):
                current_time = datetime.now(self.local_timezone).timestamp()
                remaining_cooldown = self.cool_off - (current_time - self.last_fetched_times.get(symbol, 0))

                if remaining_cooldown > 0:
                    logger.info(f"Cooldown for {symbol} is active. Remaining time: {remaining_cooldown:.2f} seconds.")
                    await asyncio.sleep(remaining_cooldown)
                    continue

                try:
                    logger.info(f"Attempt {attempt + 1} to fetch klines for {symbol}")

                    if not self.client:
                        logger.error(f"Client not initialized during klines fetch for {symbol}")
                        return pd.DataFrame()  # Return an empty DataFrame

                    klines = await self.client.futures_klines(
                        symbol=symbol,
                        interval=AsyncClient.KLINE_INTERVAL_1MINUTE,
                        start_str=self.start_time,
                        end_str=self.end_time,
                        limit=self.max_requests,
                    )

                    if not isinstance(klines, list) or len(klines) == 0:
                        logger.warning(f"No klines data returned for {symbol}, skipping.")
                        return pd.DataFrame()  # Return an empty DataFrame

                    # Create DataFrame
                    df = pd.DataFrame(klines, columns=[
                        'open_time', 'open', 'high', 'low', 'close', 'volume',
                        'close_time', 'quote_asset_volume', 'number_of_trades',
                        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
                    ])
                    df = df[['open_time', 'open', 'high', 'low', 'close', 'volume']]

                    logger.debug(f"Raw DataFrame before cleaning for {symbol}: {df.head()}")

                    # Clean DataFrame
                    df = await self.clean_data(df, symbol)

                    if isinstance(df, pd.DataFrame) and not df.empty:
                        logger.info(f"Cleaned DataFrame for {symbol}: {df.shape}")

                        # Calculate moving averages if trader is initialized
                        if self.moving_average_trader:
                            df = await self.moving_average_trader.calculate_tema_and_ma_async(
                                symbols=self.symbol,  # Use self.symbol directly
                                df={symbol: df},  # Wrap df in a dictionary
                                window_tema=[9, 20],
                                ma_windows=[7, 14, 30],
                                confirmation_intervals=[10, 3, 15],
                                confirmation_checks=[5, 20, 1]
                            )
                            if df is None:
                                logger.warning(f"Moving averages calculation returned None for {symbol}.")
                        else:
                            logger.error("MovingAverageTrader is not initialized.")

                        dataframes.append(df)
                        self.fetched_symbols.add(symbol)
                    else:
                        logger.warning(f"Cleaned DataFrame is empty or contains NaN values for {symbol}.")

                except BinanceAPIException as e:
                    logger.error(f"Binance API exception for {symbol}: {e}. Retrying...")
                    await asyncio.sleep(self.cool_off)
                except BinanceRequestException as e:
                    logger.error(f"Request exception for {symbol}: {e}. Retrying...")
                    await asyncio.sleep(self.cool_off)
                except Exception as e:
                    logger.error(f"General exception for {symbol}: {e}. Retrying...")
                    await asyncio.sleep(self.cool_off)

            logger.warning(f"Max attempts reached for {symbol}. Skipping klines fetch.")
            return pd.DataFrame()  # Return an empty DataFrame

        try:
            tasks = [fetch_single_kline(symbol) for symbol in self.symbol]
            klines_results = await asyncio.gather(*tasks, return_exceptions=True)

            for symbol, klines in zip(self.symbol, klines_results):
                if isinstance(klines, pd.DataFrame) and not klines.empty:
                    dataframes.append(klines)

            if dataframes:
                return pd.concat(dataframes, ignore_index=True)
            else:
                logger.warning("No valid dataframes to concatenate, returning empty DataFrame.")
                return pd.DataFrame()

        finally:
            await self.check_session_inactivity(last_active_time)
    
    async def initialize_symbols(self):
        from main import read_symbols_from_file
        """Ensure symbols are initialized before fetching data."""
        if not self.symbol:
            self.symbol = await read_symbols_from_file('symbols.txt')
        if not self.symbol:
            raise ValueError("No symbols found to initialize.")

    async def fetch_live_klines(self):
        from main import read_symbols_from_file
        """Connects to Binance WebSocket for live Kline data."""

        # Load symbols from file or another source and validate the result
        self.symbols = await read_symbols_from_file(filename="T:\\Trading BOT Crypto - Profit\\symbols_orders_initializations.txt")

        # Validate that self.symbols is a list of strings
        if not isinstance(self.symbols, list):
            logging.error("Expected self.symbols to be a list, but got something else.")
            return

        if not all(isinstance(symbol, str) for symbol in self.symbols):
            logging.error("self.symbols should contain only strings.")
            return

        if not self.symbols:
            logging.error("No valid symbols available for the WebSocket connection.")
            return

        # Create combined stream string for WebSocket
        stream_names = [f"{symbol.lower()}@kline_1m" for symbol in self.symbols]
        combined_stream = "/".join(stream_names)
        url = f"wss://fstream.binance.com/stream?streams={combined_stream}"

        logging.debug(f"Connecting to WebSocket URL: {url}")

        while True:
            try:
                async with websockets.connect(url) as websocket:
                    logging.info(f"Connected to WebSocket for symbols: {self.symbols}")
                    start_time = datetime.now()

                    while True:
                        try:
                            response = await websocket.recv()
                            kline_data = json.loads(response)

                            # Log the data to inspect its structure
                            logging.debug(f"Received WebSocket data: {kline_data}")

                            if 'data' in kline_data:
                                if 'k' in kline_data['data']:
                                    kline_info = kline_data['data']['k']  # 'k' contains the actual kline data

                                    # Validate that the expected keys are present in the data
                                    if all(key in kline_info for key in ['t', 'o', 'h', 'l', 'c', 'v']):
                                        kline_fields = {
                                            'timestamp': kline_info['t'],  # Timestamp
                                            'open': float(kline_info['o']),  # Opening price
                                            'high': float(kline_info['h']),  # Highest price
                                            'low': float(kline_info['l']),  # Lowest price
                                            'close': float(kline_info['c']),  # Closing price
                                            'volume': float(kline_info['v'])  # Volume
                                        }

                                        # Create DataFrame from the extracted fields
                                        raw_df = pd.DataFrame([kline_fields])

                                        # Clean and process the data
                                        cleaned_df = await self.clean_data(raw_df, kline_data['stream'])

                                        # Calculate TEMA and MA values
                                        calculated_df = await self.moving_average_trader.calculate_tema_and_ma_async(
                                            symbols=self.symbols,  # Use the list of symbols
                                            df={symbol: cleaned_df for symbol in self.symbols},  # Create a dict with each symbol
                                            window_tema=[9, 20],
                                            ma_windows=[7, 14, 30],
                                            confirmation_intervals=[10, 3, 15],
                                            confirmation_checks=[5, 20, 1]
                                        )

                                        # Extract moving averages, filtering out None values
                                        ma_values = [value for window in [7, 14, 30] if (value := calculated_df.get(f'MA_{window}')) is not None]

                                        # Now use ma_values for further processing
                                        await self.process_kline_data(kline_data, ma_values)

                                    else:
                                        logging.error(f"Missing expected kline keys in data: {kline_info}")
                                else:
                                    logging.error(f"'k' key missing in kline data: {kline_data['data']}")
                            else:
                                logging.error(f"Unexpected WebSocket data format: {kline_data}")

                            # Reset connection after 24 hours
                            if datetime.now() - start_time >= timedelta(hours=24):
                                logging.info("Resetting WebSocket connection after 24 hours.")
                                break

                        except json.JSONDecodeError as json_error:
                            logging.error(f"JSON decode error: {json_error}. Response: {response}")
                            await asyncio.sleep(5)

                        except Exception as kline_error:
                            logging.error(f"Kline processing error: {kline_error}. Reconnecting...")
                            await asyncio.sleep(5)
                            break
            except websockets.ConnectionClosedError as websocket_error:
                logging.error(f"WebSocket connection closed: {websocket_error}. Reconnecting...")
                await asyncio.sleep(5)

            except Exception as general_error:
                logging.error(f"WebSocket connection error: {general_error}. Retrying...")

            await asyncio.sleep(1)  # Prevent tight loop on failure


    async def process_message(self, message):
        # Parse the message (assuming it's JSON formatted)
        data = json.loads(message)

        # Assuming the message contains the necessary information
        kline_data = {
            'open_time': data['k']['t'],
            'open': data['k']['o'],
            'high': data['k']['h'],
            'low': data['k']['l'],
            'close': data['k']['c'],
            'volume': data['k']['v'],
            'close_time': data['k']['T'],
            'quote_asset_volume': data['k']['q'],
            'number_of_trades': data['k']['n'],
            'taker_buy_base_asset_volume': data['k']['V'],
            'taker_buy_quote_asset_volume': data['k']['Q'],
            'ignore': data['k']['B']
        }

        # Create DataFrame from the kline_data
        df = pd.DataFrame([kline_data])  # Ensure it's a list of dicts

        # Clean the DataFrame
        cleaned_df = await self.clean_data(df, "ETHUSDT")
        print(cleaned_df)  # Or perform additional actions with cleaned_df
    # In process_kline_data method
    async def process_kline_data(self, kline_data, ma_values: List[Union[int, float]]):
        """Processes incoming kline data from WebSocket."""
        kline = kline_data['k']

        if kline['x']:  # If the Kline is closed
            symbol = kline['s']
            open_time = datetime.fromtimestamp(kline['t'] / 1000, tz=self.local_timezone)
            open_price = float(kline['o'])
            high_price = float(kline['h'])
            low_price = float(kline['l'])
            close_price = float(kline['c'])
            volume = float(kline['v'])

            # Append incoming kline data to buffer
            self.kline_buffer.append({
                'open_time': open_time,
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': volume
            })

            # Check if the buffer has reached the required size
            if len(self.kline_buffer) >= self.buffer_size:
                # Create DataFrame from the buffered Kline data
                df = pd.DataFrame(self.kline_buffer)

                logger.debug(f"Received Kline data for {symbol}: {df}")

                # Clean and process the DataFrame
                df = await self.clean_data(df, symbol)

                if isinstance(df, pd.DataFrame) and not df.empty:
                    logger.info(f"Cleaned DataFrame for {symbol}: {df.shape}")

                    if self.moving_average_trader is None:
                        logger.error("MovingAverageTrader is not initialized.")
                        return

                    # Use the provided MA values
                    ma_windows = ma_values[:3]  # MA windows for 7, 14, and 30
                    window_tema = ma_values[3:]  # TEMA windows for TEMA_9 and TEMA_20

                    # Ensure the DataFrame has enough data for calculations
                    if len(df) < max(max(ma_windows), max(window_tema)):
                        logger.warning(f"Not enough data to calculate moving averages for {symbol}. Required: {max(max(ma_windows), max(window_tema))}, Available: {len(df)}")
                        return  # Early return if there's insufficient data

                    # Log extracted kline values for clarity
                    logger.info(f"Kline Data - Symbol: {symbol}, Open: {open_price}, High: {high_price}, Low: {low_price}, Close: {close_price}, Volume: {volume}")

                    df = await self.moving_average_trader.calculate_tema_and_ma_async(
                        symbols=self.symbol,
                        df={symbol: df},
                        window_tema = [int(x) for x in window_tema],  # Ensure all values are integers
                        ma_windows = [int(x) for x in ma_windows]
                        # confirmation_intervals=[10, 3, 15],
                        # confirmation_checks=[5, 20, 1]
                    )

                # Clear the buffer after processing
                self.kline_buffer.clear()
            else:
                logger.debug(f"Buffered {len(self.kline_buffer)} klines for {symbol}, waiting for more data...")

    async def check_session_inactivity(self, last_active_time):
        """
        Check if the client session has been inactive for more than 3 minutes,
        and reinitialize the client if necessary.
        """
        current_time = datetime.now(self.local_timezone).timestamp()
        inactivity_duration = current_time - last_active_time

        if inactivity_duration > 180:  # 3 minutes in seconds
            print("Session inactive for more than 3 minutes, reinitializing client...")
            # await self._initialize_client(self.api_key, api_secret)
        else:
            print(f"Session active within the last {inactivity_duration:.2f} seconds, keeping client session open.")

    async def clean_data(self, df: pd.DataFrame, symbol) -> pd.DataFrame:
        max_attempts = 3
        initial_shape = df.shape
        print(f"Initial DataFrame shape for {symbol}: {initial_shape}")

        # Define moving average values
        ma_values = [7, 14, 20]  # MA periods
        tema_values = [9, 20]    # TEMA periods

        # Define numeric columns
        numeric_cols = ['close', 'open', 'high', 'low', 'volume']

        # Handle NaNs by filling them instead of dropping
        if df[numeric_cols].isnull().values.any():
            print(f"NaN values found in DataFrame for {symbol}, filling with 0.")
            df[numeric_cols] = df[numeric_cols].fillna(0)

        for attempt in range(1, max_attempts + 1):
            try:
                # Convert columns to numeric, handle conversion errors
                df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

                # Remove duplicates
                df.drop_duplicates(inplace=True)
                print(f"Shape after removing duplicates for {symbol}: {df.shape}")

                # Remove invalid values (negative prices or volume)
                invalid_mask = (df[numeric_cols] < 0).any(axis=1)
                if invalid_mask.any():
                    print(f"Invalid data found in the following rows for {symbol}:\n{df[invalid_mask]}")
                    df = df[~invalid_mask]
                    print(f"Shape after removing invalid rows for {symbol}: {df.shape}")

                # Check the number of valid rows
                if len(df) < 20:  # Adjust threshold for minimum rows based on the highest MA period
                    print(f"Not enough valid rows after cleaning for {symbol}: {len(df)}. Returning empty DataFrame.")
                    return pd.DataFrame()  # Return an empty DataFrame if not enough rows

                # Ensure the necessary columns are present before returning
                if 'timestamp' not in df.columns:
                    df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')  # Assuming open_time is in milliseconds
                if 'symbol' not in df.columns:
                    df['symbol'] = symbol  # Ensure the symbol column is present

                final_shape = df.shape
                print(f"Final DataFrame shape after cleaning for {symbol}: {final_shape}")

                # Ensure DataFrame has enough data for moving averages
                if final_shape[0] < max(ma_values):  # Check against the largest MA requirement
                    print(f"Insufficient data for moving averages for {symbol}. Required: {max(ma_values)}, Available: {final_shape[0]}")
                    return pd.DataFrame()  # Return an empty DataFrame if not enough data for moving averages

                # Returning cleaned DataFrame
                return df

            except Exception as e:
                print(f"Error during cleaning for {symbol}: {e}. Retrying {attempt}/{max_attempts}...")
                await asyncio.sleep(10)  # Optional: wait before retrying

        # If max attempts exceeded, return DataFrame with whatever data is cleaned
        return df


    async def process_data(self) -> Optional[Tuple[List[float], pd.DataFrame]]:
        from .Fetching_and_calculating_moving_averages_and_TEMA import MovingAverageTrader

        # Check if a signal has been triggered
        if self.signal_trigger is not None:
            logger.info(f"Signal triggered for {self.symbol}, stopping data fetching.")
            return [], pd.DataFrame()  # Return empty list and empty DataFrame

        max_retries = 5
        backoff_time = 20
        ma_trader = MovingAverageTrader(self.client, self.symbol)

        for attempt in range(max_retries):
            try:
                klines = await self.fetch_live_klines()  # This is expected to return DataFrame or None

                # Handle the case where no data is returned
                if klines is None:
                    logger.error(f"No klines data returned for {self.symbol} on attempt {attempt + 1}/{max_retries}.")
                    await asyncio.sleep(backoff_time)
                    backoff_time *= 1.5
                    continue
                
                # Validate klines data format
                if not isinstance(klines, pd.DataFrame) or 'close' not in klines.columns:
                    logger.error(f"Invalid klines data for {self.symbol}. Must be a DataFrame with 'close' column.")
                    await asyncio.sleep(20)
                    return None

                # Clean the data and log its shape
                df = await self.clean_data(klines, self.symbol)
                logger.info(f"Cleaned DataFrame for {self.symbol} has shape: {df.shape}")

                # Initialize MovingAverageTrader client
                if not await ma_trader.initialize_async_client(client=self.client):
                    logger.error(f"Failed to initialize MovingAverageTrader client for {self.symbol}")
                    return None

                # Calculate TEMA for specific windows
                window_temas = [9, 20]
                tema_calculation_success = await ma_trader.calculate_tema(df, window_temas)

                # Check if TEMA calculation was successful
                if not tema_calculation_success:
                    logger.error(f"TEMA calculation failed for {self.symbol}.")
                    await asyncio.sleep(20)
                    return None

                # Calculate the moving averages
                ma_values = await self.initialize_moving_average_trader()

                if ma_values is None:  # Handle None case
                    logger.error(f"Failed to calculate MAs for {self.symbol}.")
                    return None

                logger.info(f"Calculated MAs and TEMA values for {self.symbol}: {ma_values}")

                return ma_values, df  # Return both MA values and DataFrame

            except Exception as e:
                logger.error(f"Error during TEMA and MA calculation for {self.symbol}: {str(e)}")
                await asyncio.sleep(20)

        logger.error(f"Failed to fetch klines for {self.symbol} after {max_retries} attempts.")
        return None


    # def calculate_moving_averages(self, df: pd.DataFrame) -> List[float]:
    #     """Calculate moving averages from the cleaned DataFrame."""
    #     try:
    #         ma_values = [
    #             df['close'].rolling(window=7).mean().iloc[-1],
    #             df['close'].rolling(window=14).mean().iloc[-1],
    #             df['close'].rolling(window=30).mean().iloc[-1],
    #             df['TEMA_9'].iloc[-1],
    #             df['TEMA_20'].iloc[-1]
    #         ]
    #         return ma_values
    #     except Exception as e:
    #         logger.error(f"Error calculating moving averages: {str(e)}")
    #         return []  # Return a list of None if there's an error