# main
import os
import logging
import asyncio
import time
import aiofiles
import pandas as pd
import re
import traceback
# import aiodns
import dotenv
from dotenv import load_dotenv
from binance.client import AsyncClient
from dataclasses import dataclass, field
from collections import namedtuple
from decimal import Decimal

# from binance.exceptions import BinanceAPIException
# import signal
# from data.Number_of_Requests import RequestLogger 

from logger_config import get_logger
from binance.enums import Enum
from datetime import datetime, timedelta
from colorama import Fore, Style, Back
# import random
# from  aiohttp import  ClientError, ServerTimeoutError, ClientSession
from typing import List, Tuple, Callable, Optional, Literal, Sequence, Dict, Any, Union, Coroutine, Awaitable
# from data.TEMA_Calculations.Calculating_TripleExponentialMovingAverage import calculating_TEMA_async
# Import constants and custom modules
from data.config import MAX_REQUESTS, TIME_WINDOW
from data.AccountManager.AccountManager import AccountManager
from utils import TradingUtils
from data.BinanceManagerHandler import BinanceShortManager
from Order_handlers.OrderPlacement import OrderPlacement
from data.OrderManagerTakeProfit import OrderManagerTakeProfit
# from data.ProfitLossMonitor import ProfitLossMonitor
# from data.PositionSizer import PositionSizer
from data.StopLossOrders import StopLossOrders
from data.TradeHistory import TradeHistory
from data.RateLimiter import RateLimiter
# from strategy_long_activation_trades import TradingStrategy  # Update import if necessary
# from data.candles import TradingAnalyzer
# from monitoring.monitor_positions import PositionMonitor
from data.Fetching_and_calculating_moving_averages_and_TEMA import MovingAverageTrader  # Ensure this import is correct
from binance.client import Client
# from indicators_calculations.confirmation_checks import ConfirmationChecks
# current_time = datetime.datetime.now().time()


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

dotenv.load_dotenv()  # Load environment variables from a .env file

# Set the event loop policy to SelectorEventLoop for compatibility with aiodns
if os.name == 'nt':  # Check if the OS is Windows
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Initialize the Binance client with appropriate API keys
api_key = os.getenv('BINANCE_API_KEY')
api_secret = os.getenv('BINANCE_API_SECRET')
# client = AsyncClient.create(api_key, api_secret)

dotenv.load_dotenv()  # Load environment variables from a .env file

# Create an instance of RateLimiter with defined constants
rate_limiter = RateLimiter(MAX_REQUESTS, TIME_WINDOW)
logger = get_logger(__name__, blue_background=True)

MAX_RETRIES = 100
RETRY_DELAY = 5  # seconds
# Define a global dictionary to store prices
CHECK_INTERVAL = 30  # seconds

prices = {}

# Update this line to use the correct path
script_dir = os.path.dirname(os.path.realpath(__file__))

# request_logger = RequestLogger("binance_requests_log.txt")


# Create an instance of LoggerAdapter with the custom attribute


async def confirmation_loop(symbol: List[str], check_function, intervals=1, max_checks=5, edge_case=False):
    """
    Loop to perform confirmation checks at specified intervals.

    :param symbol: Trading pair symbol.
    :param check_function: Function to check conditions.
    :param intervals: Interval between checks in seconds.
    :param max_checks: Maximum number of successful checks required.
    :param edge_case: Flag to handle edge case monitoring.
    :return: True if conditions are confirmed, False otherwise.
    """
    if intervals <= 0:
        logger.error("Invalid intervals value. Must be greater than 0.")
        await asyncio.sleep(20)  # Brief sleep to avoid tight loop in case of consistent errors
        return False

    # Ensure max_checks is at least 1 and has a sensible default value
    if max_checks <= 0:
        max_checks = 20

    consecutive_checks = 0
    start_time = datetime.now()
    edge_case_triggered = False

    while consecutive_checks < max_checks:
        try:
            condition_met = await check_function()
            if condition_met:
                consecutive_checks += 1
                logger.info(f"symbol {symbol}: Condition met. Confirmation check {consecutive_checks}/{max_checks}")
                await asyncio.sleep(5.0)
            else:
                consecutive_checks = 0
                logger.info(f"symbol {symbol}: Condition failed, resetting checks.")
                await asyncio.sleep(5.0)
                # Reset edge case flag
                edge_case_triggered = False

                if edge_case:
                    elapsed_time = (datetime.now() - start_time).total_seconds()
                    logger.debug(f"Elapsed time for edge case check: {elapsed_time:.2f} seconds")
                    if elapsed_time > 600 and not edge_case_triggered:
                        logger.info(f"symbol {symbol}: Edge case triggered. Conditions not met for more than 10 minutes.")
                        await asyncio.sleep(5.0)
                        edge_case_triggered = True
                        end_edge_case_time = datetime.now() + timedelta(minutes=5)
                        while datetime.now() < end_edge_case_time:
                            if await check_function():
                                logger.info(f"symbol {symbol}: Candlestick price condition passed during edge case.")
                                return True
                            else:
                                logger.info(f"symbol {symbol}: Conditions no longer valid during edge case monitoring.")
                            await asyncio.sleep(5)  # Keep checking while in edge case
                        return False  # Exit if edge case monitoring fails

            await asyncio.sleep(intervals)

        except Exception as e:
            logger.error(f"Error during confirmation loop for {symbol}: {e}")
            await asyncio.sleep(20)  # Brief sleep to avoid tight loop in case of consistent errors

    return consecutive_checks >= max_checks

async def monitor_positions(symbol: List[str], entry_timestamp, interval=2):
    """
    Monitor positions for a given symbol and log the duration the position has been open.

    :param symbol: The trading pair symbol (e.g., 'ETHUSDT').
    :param entry_timestamp: The timestamp when the position was entered.
    :param interval: Interval in seconds between monitoring checks.
    """
    if interval <= 0:
        logger.error("Invalid monitoring interval. Must be greater than zero.")
        await asyncio.sleep(20)  # Brief sleep to avoid tight loop in case of consistent errors

        return

    try:
        logger.info(f"Monitoring position for {symbol}...")
        await asyncio.sleep(5.0)
        while True:
            try:
                current_timestamp = datetime.now()
                duration = current_timestamp - entry_timestamp
                logger.info(f"Position for {symbol} has been open for: {duration}")
                await asyncio.sleep(5.0)
                await asyncio.sleep(interval)
            except Exception as e:
                logger.error(f"Error during position monitoring for {symbol}: {e}")
                await asyncio.sleep(20)  # Brief sleep to avoid tight loop in case of consistent errors
                await asyncio.sleep(interval)  # Wait before retrying
    except Exception as e:
        logger.error(f"Unexpected error monitoring positions for {symbol}: {e}")
        await asyncio.sleep(20)  # Brief sleep to avoid tight loop in case of consistent errors



async def log_request(logger, client, method, endpoint, **kwargs):
    # Log the request
    logger.log_request(method.upper(), endpoint)

    # Execute the actual API call (assuming client is Binance API client)
    if method.upper() == 'GET':
        response = await client.get(endpoint, **kwargs)
    elif method.upper() == 'POST':
        response = await client.post(endpoint, **kwargs)
    else:
        raise ValueError("Unsupported request type")
    
    return response

# async def process_data_after_fetching_k_lines(data: pd.DataFrame) -> Optional[pd.DataFrame]:
#     try:
#         if data is None or data.empty:
#             logger.warning("DataFrame is empty or None.")
#             return None

#         columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
#         data = pd.DataFrame(data, columns=columns)

#         logger.info(f"Data shape before cleaning: {data.shape}")

#         # Clean Data
#         for col in columns[1:]:
#             data[col] = pd.to_numeric(data[col], errors='coerce')
#         data.dropna(inplace=True)

#         logger.info(f"Data shape after cleaning: {data.shape}")

#         if data.empty:
#             logger.warning("DataFrame is empty after cleaning.")
#             valid, message = await TradingUtils.validate_data(data)
#             if not valid:
#                 logger.error(f"Validation failed: {message}")
#                 return None
#             logger.info("Data has been successfully validated and fetched.")

#         # Calculate TEMA and Moving Averages
#         window_tema = 9
#         data['ema'] = data['close'].ewm(span=window_tema, adjust=False).mean()
#         data['ema_ema'] = data['ema'].ewm(span=window_tema, adjust=False).mean()
#         data['ema_ema_ema'] = data['ema_ema'].ewm(span=window_tema, adjust=False).mean()
#         data['TEMA'] = 3 * (data['ema'] - data['ema_ema']) + data['ema_ema_ema']

#         ma_windows = [7, 14, 30]
#         for window in ma_windows:
#             data[f'MA_{window}'] = data['close'].rolling(window=window).mean()

#         required_columns = ['TEMA'] + [f'MA_{window}' for window in ma_windows]
#         missing_columns = [col for col in required_columns if col not in data.columns]

#         if missing_columns:
#             logger.error(f"Missing columns: {missing_columns}.")
#             return None

#         return data

#     except KeyError as ke:
#         logger.error(f"Key error: {str(ke)}", exc_info=True)
#     except ValueError as ve:
#         logger.error(f"Value error: {str(ve)}", exc_info=True)
#     except Exception as e:
#         logger.error(f"An unexpected error occurred: {str(e)}", exc_info=True)
#     await asyncio.sleep(1)
#     return None
from data.Fetching_historical_OHCL_Values import KlineDataProcessor

@dataclass
class Services:
    moving_average_confirmation_trader: MovingAverageTrader
    order_placement_position: OrderPlacement
    stop_loss_calculations: StopLossOrders
    bsm_Binance_short_manager: BinanceShortManager
    order_manager_take_profit: OrderManagerTakeProfit
    k_lines_configuration: KlineDataProcessor
    moving_average_trader: MovingAverageTrader  # Include this in Services

    def __init__(
        self,
        moving_average_confirmation_trader: MovingAverageTrader,
        order_placement_position: OrderPlacement,
        stop_loss_calculations: StopLossOrders,
        bsm_Binance_short_manager: BinanceShortManager,
        order_manager_take_profit: OrderManagerTakeProfit,
        k_lines_configuration: KlineDataProcessor,
        moving_average_trader: MovingAverageTrader  # Add this parameter
    ):
        self.moving_average_confirmation_trader = moving_average_confirmation_trader
        self.order_placement_position = order_placement_position
        self.stop_loss_calculations = stop_loss_calculations
        self.bsm_Binance_short_manager = bsm_Binance_short_manager
        self.order_manager_take_profit = order_manager_take_profit
        self.k_lines_configuration = k_lines_configuration
        self.moving_average_trader = moving_average_trader  # Assign it to the class





class ProcessSymbol:
    def __init__(self, symbol, client, api_key=None, api_secret=None):
        self.symbol = symbol
        self.client = client
        self.api_key = api_key or os.getenv('BINANCE_API_KEY')
        self.api_secret = api_secret or os.getenv('BINANCE_API_SECRET')
        self.logger = logging.getLogger(__name__)

        # Initialize other classes
        self.account_manager = AccountManager(client, self.api_key, self.api_secret)
        self.utilities_helper_functions = TradingUtils(self.client, self.symbol)
        self.binance_short_manager = BinanceShortManager(self.client, symbol, api_key, api_secret)
        self.order_placement_instance = OrderPlacement(self.client, self.symbol)
        self.order_manager_take_profit = OrderManagerTakeProfit(self.client, self.symbol)
        self.stop_loss_orders = StopLossOrders(self.client, self.symbol)
        self.trade_history = TradeHistory(trade_history_file='trade_history.txt')
        self.moving_average_trader = MovingAverageTrader(self.client, self.symbol)
        
        # Initialize KlineDataProcessor with the correct file path
        self.kline_data_processor = KlineDataProcessor(
            client=self.client,
            symbols_file_path="T:\\Trading BOT Crypto - Profit\\symbols_orders_initializations.txt",
            max_requests=1000,
            start_time="2024-08-29", 
            end_time="2100-01-01", 
            fetch_attempts=3, 
            cool_off=60, 
            stop_if_fetched=False, 
            log_retries=True, 
            signal_trigger=None,
        )

        # Initialize Services dataclass with the correct instances
        self.services = Services(
            moving_average_confirmation_trader=self.moving_average_trader,
            order_placement_position=self.order_placement_instance,
            stop_loss_calculations=self.stop_loss_orders,
            bsm_Binance_short_manager=self.binance_short_manager,
            order_manager_take_profit=self.order_manager_take_profit,
            k_lines_configuration=self.kline_data_processor,
            moving_average_trader=self.moving_average_trader
        )



    async def initialize_client(self, api_key: str, api_secret: str):
        """Initialize the client and dependent classes."""
        try:
            self.client = await AsyncClient.create(api_key, api_secret)

            # Re-initialize other classes now that the client is available
            self.account_manager = AccountManager(self.client, api_key, api_secret)
            self.utilities_helper_functions = TradingUtils(self.client, self.symbol)
            self.binance_short_manager = BinanceShortManager(self.client, self.symbol, api_key, api_secret)
            self.order_placement_instance = OrderPlacement(self.client, self.symbol)
            self.order_manager_take_profit = OrderManagerTakeProfit(self.client, self.symbol)
            self.stop_loss_orders = StopLossOrders(self.client, self.symbol)
            self.moving_average_trader = MovingAverageTrader(self.client, self.symbol)

            # Re-initialize KlineDataProcessor if necessary
            self.kline_data_processor = KlineDataProcessor(
                client=self.client,
                symbols_file_path="T:\\Trading BOT Crypto - Profit\\symbols_orders_initializations.txt",
                max_requests=1000,
                start_time="2024-08-29",
                end_time="2100-01-01",
                fetch_attempts=3,
                cool_off=60,
                stop_if_fetched=False,
                log_retries=True,
                signal_trigger=None,
            )

            self.logger.info("Client and dependent classes initialized successfully.")
        
        except Exception as e:
            self.logger.error(f"An error occurred during initialization: {e}")

        # Update Services dataclass with new instances
        self.services = Services(
            moving_average_confirmation_trader=self.moving_average_trader,
            order_placement_position=self.order_placement_instance,
            stop_loss_calculations=self.stop_loss_orders,
            bsm_Binance_short_manager=self.binance_short_manager,
            order_manager_take_profit=self.order_manager_take_profit,
            k_lines_configuration=self.kline_data_processor,
            moving_average_trader=self.moving_average_trader
        )



        # Optionally log successful initialization of components
        self.logger.info(f"Client and components initialized for {self.symbol}")

    async def check_short_conditions(self, symbol, _ohlcv_df: Optional[Dict[str, pd.DataFrame]]) -> bool:
        maximum_number_of_retries = 3
        retries = 0

        while retries < maximum_number_of_retries:
            try:
                logger.info(f"Attempt {retries + 1}/{maximum_number_of_retries} for checking conditions for {symbol}.")
                await asyncio.sleep(5.0)

                success = await self.moving_average_trader.calculate_tema_and_ma_async(
                    symbols=[symbol],
                    df=_ohlcv_df,  
                    window_tema=[9, 20],
                    ma_windows=[7, 14, 30]
                )

                if not success:
                    logger.warning(f"Condition checks did not succeed for symbol {symbol}. Retrying after backoff...")
                    await asyncio.sleep(10.0)
                    retries += 1
                    await asyncio.sleep(2 ** retries)
                    continue

                candle_df = await self.kline_data_processor.fetch_live_klines()
                if candle_df is None or candle_df.empty:
                    logger.warning(f"Fetched candlestick data is empty for {symbol}.")
                    await asyncio.sleep(10)
                    return False

                check_duration = 60  # Ensure this is defined
                if not await self.moving_average_trader.check_candlestick_prices(symbol, check_duration=check_duration):
                    logger.info(f"Candlestick conditions not met for {symbol}.")
                    await asyncio.sleep(5.0)
                    return False

                logger.info(f"Conditions met for symbol {symbol}.")
                await asyncio.sleep(5.0)
                return True

            except Exception as e:
                retries += 1
                logger.error(f"Exception occurred on attempt {retries}/{maximum_number_of_retries} while processing {symbol}: {str(e)}", exc_info=True)
                await asyncio.sleep(2 ** retries)

        logger.warning(f"Failed to confirm short signal conditions for symbol {symbol} after {maximum_number_of_retries} attempts.")
        await asyncio.sleep(10.0)
        return False

    async def check_first_condition(self, symbol: List[str], df: pd.DataFrame) -> bool:
        try:
            if df is None or df.empty:
                self.logger.warning(f"DataFrame is empty for {symbol}.")
                return False

            tema_20_value = df['TEMA_20'].iloc[-1]
            tema_9_value = df['TEMA_9'].iloc[-1]
            check_duration = 60

            if tema_20_value > tema_9_value:
                self.logger.info(f"TEMA 20 > TEMA 9 for {symbol}. Condition met.")
                return True
            else:
                self.logger.info(f"TEMA 20 <= TEMA 9 for {symbol}. Checking candlestick prices...")
                return await self.moving_average_trader.check_candlestick_prices(symbol, check_duration)
        except Exception as e:
            self.logger.error(f"Error in check_first_condition for {symbol}: {str(e)}")
            return False







    
    # async def close_client(self):
    #     if self.client and self.client.session:
    #         await self.client.session.close()
    #         self.client.session = None


    async def process_symbol_loop(self, symbol, client=None, max_iterations=10000000):
        retry_attempts = 10
        retry_delay = 5
        cooldown_period = 60  # Cooldown period after each full iteration (in seconds)

        if client is None:
            self.logger.info(f"Initializing Binance client for {symbol}.")
            client = await self.account_manager.initialize_client()
            if not client:
                self.logger.error("Failed to initialize Binance client.")
                return

        for iteration in range(max_iterations):
            try:
                self.logger.info(f"Processing symbol: {symbol}")

                # Fetch Wallet Information
                await asyncio.sleep(5.0)  # Simulate delay in fetching
                wallet_info = await self.account_manager.fetch_futures_balance(asset='BNFCR')
                self.logger.info(f"Wallet information fetched successfully: {wallet_info}")

                # Fetch Klines Data and Process it
                k_lines_fetched = await self.kline_data_processor.fetch_live_klines()

                if not isinstance(k_lines_fetched, dict) or symbol not in k_lines_fetched:
                    raise ValueError(f"No klines data returned for symbol: {symbol}.")

                self.logger.info(f"Fetched Klines: {k_lines_fetched}")

                result = await self.kline_data_processor.process_data()
                if result is None:
                    raise ValueError(f"Failed to process data for {symbol}. Received None.")

                ma_values_list, ohlcv_df = result

                # Initialize MovingAverageTrader only after processing data
                moving_average_trader = MovingAverageTrader(client, symbol)
                await moving_average_trader.initialize_async_client(client)  # Ensure it is initialized

                tema_values = await moving_average_trader.calculate_tema(symbol, [9, 20])
                self.logger.info(f"Moving Averages and Data processed successfully for {symbol}.")

                # Check Trading Conditions AFTER calculating MA and TEMA
                conditions_met = await moving_average_trader.orchestrate_short_conditions(
                    symbol=symbol,
                    df=ohlcv_df,  # Pass DataFrame
                    ma_values=ma_values_list  # Pass List[float]
                )

                if conditions_met:
                    # Perform Risk Management Checks before Opening Position
                    balance_sufficient = await self.account_manager.fetch_futures_balance(client)
                    maintenance_margin_ok = await self.account_manager.fetch_futures_balance(client)

                    if balance_sufficient and maintenance_margin_ok:
                        await self.order_placement_instance.open_short_position(symbols=symbol, entry_price=None)
                        self.logger.info(f"Short position opened for {symbol}")

                        # Fetch and log the new positions
                        await self.order_placement_instance.fetch_running_positions()
                        self.logger.info(f"New short position fetched and logged for {symbol}")

                self.logger.info(f"Completed processing loop for {symbol}. Sleeping for 10 seconds before next iteration.")
                await asyncio.sleep(10.0)

                # Introduce Cooldown after each full iteration
                self.logger.info(f"Cooling down for {cooldown_period} seconds after processing {symbol}.")
                await asyncio.sleep(cooldown_period)

            except ValueError as ve:
                self.logger.error(f"ValueError processing symbol {symbol}: {ve}")
                retry_attempts -= 1
                await self._handle_retry(retry_attempts, symbol, retry_delay)

            except Exception as e:
                self.logger.error(f"Unexpected error processing symbol {symbol}: {e}")
                retry_attempts -= 1
                await self._handle_retry(retry_attempts, symbol, retry_delay)

            finally:
                self.logger.info(f"Exiting loop for {symbol}. Total iterations: {iteration + 1}")

            await asyncio.sleep(5.0)  # Add a delay between processing different symbols if needed


    async def _handle_retry(self, retry_attempts, symbol, retry_delay):
        if retry_attempts <= 0:
            self.logger.error(f"Max retry attempts reached for {symbol}. Exiting loop.")
            raise Exception(f"Max retry attempts reached for {symbol}.")
        await asyncio.sleep(retry_delay)




async def periodic_task(interval: int, coro: Callable[..., Coroutine[Any, Any, Any]], *args: Any):
    while True:
        await coro(*args)
        await asyncio.sleep(interval)



filename = (f"T:\\Trading BOT Crypto - Profit\\symbols_orders_initializations.txt")
# Function to read symbols from a file
async def read_symbols_from_file(filename: str) -> list:
    """
    Reads and processes a list of trading symbols from a file asynchronously.

    The function reads symbols from the provided file, where each symbol is 
    expected to be separated by commas or appear on individual lines. It strips 
    whitespace or newline characters and ensures valid symbols are returned as 
    a list. The function logs each step of the process for debugging purposes.

    Args:
        filename (str): The path to the file containing the symbols.
    
    Returns:
        list: A list of cleaned trading symbols.
    
    Workflow:
        1. Log the attempt to open the file at the specified location.
        2. Open the file asynchronously using `aiofiles` in read mode.
        3. Read all the lines from the file asynchronously.
        4. Process each line by:
            - Splitting symbols by commas if they are comma-separated.
            - Stripping whitespace, newlines, or extra spaces.
            - Ignoring empty strings or invalid symbols.
        5. Extend the list of symbols with cleaned symbols.
        6. Log the successfully loaded symbols and return the list.
    
    Notes:
        - If an exception occurs during file reading, it is logged, and 
          an empty list is returned.
        - Symbols should be formatted correctly (e.g., 'BNBUSDT', 'ETHUSDT') 
          without any extra spaces or invalid characters.

    Example file content:
        BNBUSDT, ETHUSDT, DOGEUSDT
        LTCUSDT
        XRPUSDT

    Example usage:
        symbols = await read_symbols_from_file('symbols.txt')
        print(symbols)  # Output: ['BNBUSDT', 'ETHUSDT', 'DOGEUSDT', 'LTCUSDT', 'XRPUSDT']
    """
    symbols = []
    try:
        logger.info(f"Checking file at: {filename}")
        async with aiofiles.open(filename, mode='r') as f:
            # Read all the lines from the file
            lines = await f.readlines()

            # Process each line and clean it up
            for line in lines:
                # Split by comma or strip newline/extra spaces if symbols are on separate lines
                line_symbols = line.split(',')
                symbols.extend([symbol.strip() for symbol in line_symbols if symbol.strip()])
        
        logger.info(f"Loaded symbols: {symbols}")
        return symbols
    except Exception as e:
        logger.error(f"Error reading symbols from file: {e}")
        await asyncio.sleep(20)
        return []



# Function to chunk symbols into batches
def chunk(symbols, batch_size):
    for i in range(0, len(symbols), batch_size):
        yield symbols[i:i + batch_size]


async def save_symbols_to_file(symbols, filename):
    with open(filename, 'w') as file:
        for symbol in symbols:
            file.write(f"{symbol}\n")
            break


def is_valid_symbol(symbol):
    """Check if the symbol contains only valid string characters (alphanumeric)."""
    return bool(re.match(r'^[A-Z]{2,}-[A-Z]{2,}$', symbol))


async def read_existing_symbols(filename):
    """Read existing symbols from the file."""
    if not os.path.isfile(filename):
        return set()
    
    async with aiofiles.open(filename, 'r') as file:
        contents = await file.read()
    
    existing_symbols = set(line.strip() for line in contents.splitlines())
    return existing_symbols


async def update_symbols_file(filename, new_symbols):
    """Append new symbols to the file if they are not already present."""
    existing_symbols = await read_existing_symbols(filename)
    
    symbols_to_append = [symbol for symbol in new_symbols if symbol not in existing_symbols]
    
    if symbols_to_append:
        async with aiofiles.open(filename, 'a') as file:
            await file.write('\n'.join(symbols_to_append) + '\n')
        logger.info(f"Appended new symbols to file: {symbols_to_append}")
    else:
        logger.info("No new symbols to append.")



async def write_symbols_to_file(filename, symbols):
    """Write valid symbols to the file."""
    try:
        async with aiofiles.open(filename, 'w') as file:
            await file.write('\n'.join(symbols) + '\n')
        logger.info(f"Symbols written to file: {symbols}")
    except IOError as e:
        logger.error(f"Error writing to file '{filename}': {e}")
        await asyncio.sleep(20)



async def initialize_services(
    client: AsyncClient,
    symbols_to_process: List[str],
    api_key: str,
    api_secret: str,
    symbols_file_path: str
) -> List[Services]:
    if client is None:
        logger.error("Client is None, cannot initialize services.")
        return []

    if not symbols_to_process:
        logger.error("No symbols to process.")
        return []

    logger.info(f"Symbols to process: {symbols_to_process}")

    services = []

    # Initialize the services for each symbol
    for symbol in symbols_to_process:
        try:
            logger.info(f"Initializing services for symbol: {symbol}")

            # Initialize each service instance
            moving_average_trader = MovingAverageTrader(client, symbol)
            order_placement_position = OrderPlacement(client, symbol)
            bsm_Binance_short_manager = BinanceShortManager(client, symbol, api_key, api_secret)
            stop_loss_calculations = StopLossOrders(client, symbol)
            order_manager_take_profit = OrderManagerTakeProfit(client, symbol)

            k_lines_configuration = KlineDataProcessor(
                client=client,
                moving_average_trader_class=moving_average_trader.calculate_tema_and_ma_async,
                symbols_file_path=symbols_file_path,
                max_requests=1000,
                start_time="2024-08-29",
                end_time="2100-01-01",
                fetch_attempts=3,
                cool_off=60,
                stop_if_fetched=False,
                log_retries=True,
                signal_trigger=None,
            )

            # Run some initial calculations for each service
            await asyncio.gather(
                moving_average_trader.calculate_tema_and_ma_async(symbol),
                stop_loss_calculations.fetch_latest_candle(symbol),
                order_manager_take_profit.calculate_dynamic_take_profit_short(symbol)
            )

            services.append(Services(
                k_lines_configuration=k_lines_configuration,
                bsm_Binance_short_manager=bsm_Binance_short_manager,
                moving_average_confirmation_trader=moving_average_trader,
                stop_loss_calculations=stop_loss_calculations,
                order_placement_position=order_placement_position,
                order_manager_take_profit=order_manager_take_profit,
                moving_average_trader=moving_average_trader
            ))

            logger.info(f"Services initialized successfully for {symbol}")

        except Exception as e:
            logger.error(f"Error initializing services for {symbol}: {e}", exc_info=True)

    if client:
        await client.close_connection()  # Ensure client connection is closed after service initialization

    return services





async def main():
    load_dotenv()  # Load environment variables
    api_key = os.getenv('BINANCE_API_KEY')
    api_secret = os.getenv('BINANCE_API_SECRET')

    if not api_key or not api_secret:
        logger.error("API key or secret not found in environment variables.")
        return

    client = await AsyncClient.create(api_key, api_secret)  # Await client creation
    symbols = ['ETHUSDT']
    process_symbol_instances = []  # List to hold ProcessSymbol instances
    services = {}  # Dictionary to hold all services initialized for each symbol

    for symbol in symbols:
        try:
            # Initialize ProcessSymbol class for each symbol
            process_symbol_instance = ProcessSymbol(symbol, client, api_key, api_secret)
            process_symbol_instances.append(process_symbol_instance)  # Store the instance
            
            logger.info(f"ProcessSymbol instance created for {symbol}.")

            # Fetch klines data for the symbol
            # Pass the symbol to fetch klines
            klines = await process_symbol_instance.kline_data_processor.fetch_live_klines()  
            if klines is not None and not klines.empty:
                logger.info(f"Klines fetched successfully for {symbol}.")

                # Initialize other components
                moving_average_trader = MovingAverageTrader(client, symbol)
                await moving_average_trader.calculate_tema_and_ma_async(symbol)
                logger.info(f"MovingAverageTrader initialized for {symbol}")

                # Initialize other managers (same as before)
                # ...

                # Store all initialized services for this symbol
                services[symbol] = {
                    'process_symbol_instance': process_symbol_instance,
                    'moving_average_trader': moving_average_trader,
                    # Store other services...
                }

                # Proceed to process symbol in the loop
                await process_symbol_instance.process_symbol_loop(
                    symbols,  # Pass the relevant parameters
                    klines,   # Assuming klines is a DataFrame
                )
            else:
                logger.warning(f"Failed to fetch valid klines for {symbol}")

        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}", exc_info=True)

    await client.close_connection()  # Close the client connection
    logger.info("Client connection closed.")

if __name__ == "__main__":
    asyncio.run(main())