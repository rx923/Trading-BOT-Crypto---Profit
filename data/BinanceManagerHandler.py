import os
import logging
import asyncio
import pandas as pd
import json
from typing import Optional, Any, Dict, Union, List
from decimal import Decimal
import urllib.parse
import time
import inspect
import hashlib
import hmac
import websockets
from datetime import datetime
from binance.client import AsyncClient, Client
from binance.exceptions import BinanceAPIException
from colorama import Fore, Style, Back
import colorama
from dotenv import load_dotenv
# from .PositionSizer import PositionSizer
# from data.TEMA_Calculations.Calculating_TripleExponentialMovingAverage import calculating_TEMA_async
# from .OrderManagerTakeProfit import OrderManagerTakeProfit
# from .Fetching_and_calculating_moving_averages_and_TEMA import MovingAverageTrader
from logger_config import get_logger, CustomLoggerAdapter
from .AccountManager.AccountManager import AccountManager
# from .Fetching_historical_OHCL_Values import KlineDataProcessor
from logging import LoggerAdapter
# Correct relative imports within the data directory
# from Order_handlers.OrderPlacement import OrderPlacement

logger = CustomLoggerAdapter(get_logger(__name__, blue_background=True), {'blue_background': True})




colorama.init(autoreset=True)

load_dotenv()  # Load environmental variables from .env file if present
api_key = os.getenv('BINANCE_API_KEY')
api_secret = os.getenv('BINANCE_API_SECRET')
# client = AsyncClient.create(api_key=api_key, api_secret=api_secret)


class BinanceShortManager:
    def __init__(self, client, symbol, api_key=None, api_secret=None):
        self.client = client
        self.symbol = symbol
        self.api_key = api_key or os.getenv('BINANCE_API_KEY')  # Fetch from environment if not passed
        self.api_secret = api_secret or os.getenv('BINANCE_API_SECRET')  # Fetch from environment if not passed
        self.prices = {}
        
        if not self.api_key or not self.api_secret:
            raise ValueError("API key and secret must be provided")

        self.unrealized_pnl_accumulated = None
        self._order_placement_configured = None  # Set in async initialization
        self.moving_average_trader_handling_positions_ = None  # Set in async initialization
        self.time_offset = 0
        self.account_manager = AccountManager(self.client, self.api_key, self.api_secret)
        # Initialize it as a dictionary or list as needed
        # Initialize klines_handler correctly after async setup of moving_average_trader_handling_positions_
        self.klines_handler = None



    async def initialize_client(self):
        if self.client:
            await self.close_client()  # Ensure old client is closed
        self.client = await AsyncClient.create(api_key, api_secret)
        print("Client initialized successfully.")



    @classmethod
    async def get_client(cls, api_key, api_secret):
        if cls._client_instance is None:
            cls._client_instance = await AsyncClient.create(api_key, api_secret)
        return cls._client_instance
    

    @staticmethod
    def generate_signature(params: dict, api_secret: str) -> str:
        query_string = urllib.parse.urlencode(sorted(params.items()), quote_via=urllib.parse.quote)
        return hmac.new(api_secret.encode(), query_string.encode(), hashlib.sha256).hexdigest()

    # async def fetch_historical_klines(self, symbol, interval, start_str):
    #     try:
    #         klines = await self.client.get_historical_klines(symbol, interval, start_str)
    #         return klines
    #     except BinanceAPIException as e:
    #         logging.error(f"Error fetching historical data for {symbol}: {e}")
    #         return None
    async def fetch_prices(self):
        url = 'wss://stream.binance.com:9443/stream?streams=' + '/'.join(f'{symbol.lower()}@ticker' for symbol in self.symbol)
        
        while True:
            try:
                async with websockets.connect(url) as websocket:
                    while True:
                        try:
                            response = await websocket.recv()
                            data = json.loads(response)
                            # Ensure the 'data' field is present and has expected structure
                            if 'data' in data and 's' in data['data'] and 'c' in data['data']:
                                symbol = data['data']['s']
                                price = data['data']['c']
                                self.prices[symbol] = price
                                print(f"Symbol: {symbol}, Price: {price}")
                            else:
                                print("Unexpected data format received.")
                        except json.JSONDecodeError:
                            print("Error decoding JSON data.")
                        except KeyError as e:
                            print(f"Missing expected key in data: {e}")
                        except Exception as e:
                            print(f"Error processing data: {e}")
                        # Wait for a short period before fetching new data
                        await asyncio.sleep(1)  # Shorter interval to handle rapid updates
            except (websockets.ConnectionClosed, websockets.InvalidMessage) as e:
                print(f"WebSocket error: {e}")
            except Exception as e:
                print(f"Unexpected error: {e}")
            
            # Reconnect after a short delay if the connection is lost
            await asyncio.sleep(5)  # Delay before attempting to reconnect

    async def start(self):
        # Create a task to periodically fetch prices
        fetch_task = asyncio.create_task(self.fetch_prices())
        await asyncio.sleep(3600)  # Run for an hour before stopping (adjust as needed)
        fetch_task.cancel()  # Cancel the fetch task after the period

    def update_prices(self, symbol, new_price):
        # Use 'self.prices' to refer to the instance's prices dictionary
        self.prices[symbol] = new_price

    def process_data(self, data):
        print(f"Received data: {data}")
        if 'data' in data:
            ticker_data = data['data']
            symbol = ticker_data.get('s', 'unknown_symbol')
            price = ticker_data.get('c', 'unknown_price')
            print(f"Symbol: {symbol}, Price: {price}")

    def get_current_prices_df(self):
        return pd.DataFrame(list(self.prices.items()), columns=['symbol', 'price'])
    
    
    async def sync_server_time(self):
        if self.client is None:
            logging.error("Client not initialized.")
            return
        
        try:
            server_time_response = await self.client.get_server_time()  # Fetch server time
            server_time = server_time_response['serverTime']
            client_time = int(time.time() * 1000)  # Get client time in milliseconds
            self.time_offset = server_time - client_time
            logging.info(f"Time offset updated: {self.time_offset} ms")
        except Exception as e:
            logging.error(f"Error syncing server time: {e}", exc_info=True)


    async def ensure_client_initialized(self) -> bool:
        if self.client is None:
            # Re-initialize the client if needed
            try:
                await self.initialize_client()  # Ensure this method properly initializes the client
                logging.info("Client re-initialized successfully.")
            except Exception as e:
                logging.error(f"Error initializing the client: {e}", exc_info=True)
                return False
        return True


    async def check_ongoing_positions(self, symbols: Optional[List[str]] = None) -> Dict[str, Optional[Dict]]:
        try:
            open_positions = {}

            logger.info("Fetching ongoing positions...")

            # Ensure the client is initialized and connected
            if not await self.ensure_client_initialized():
                return {}

            # Synchronize time
            await self.sync_server_time()


            # Fetch the balance for BNFCR
            try:
                balance_info = await self.client.futures_account_balance()  # type: ignore
                if not balance_info:
                    logging.error("Failed to fetch balance information.")
                    return {}

                # Find BNFCR balance and convert to float
                bnfcr_balance = next((float(item['balance']) for item in balance_info if item['asset'] == 'BNFCR'), 0.0)
                if bnfcr_balance <= 0.0:
                    logging.warning(f"BNFCR balance is zero or negative: {bnfcr_balance}")
                    return {}

                logging.info(f"Updated balance for BNFCR: {bnfcr_balance}")
            except Exception as e:
                logging.error(f"Error fetching BNFCR balance: {e}", exc_info=True)
                return {}

            # Fetch account positions
            try:
                account_info = await self.client.futures_account()  # type: ignore
                logging.debug(f"Account info fetched: {account_info}")
            except Exception as e:
                logging.error(f"Error fetching ongoing positions: {e}", exc_info=True)
                return {}

            # Process the positions data
            for position in account_info.get('positions', []):
                try:
                    symbol = position.get('symbol')

                    # Check if the symbol is in the list (if symbols are provided)
                    if symbols and symbol not in symbols:
                        continue  # Skip symbols not in the provided list

                    position_amt = float(position.get('positionAmt', 0.0))

                    if position_amt != 0.0:  # Check for open positions
                        entry_price = float(position.get('entryPrice', 0.0))
                        entry_time = datetime.fromtimestamp(int(position.get('updateTime', 0)) / 1000.0)
                        unrealized_pnl = await self.get_unrealized_pnl(symbol)

                        open_positions[symbol] = {
                            'entry_time': entry_time,
                            'initial_size': position_amt,
                            'unrealized_pnl': unrealized_pnl,
                            'entry_price': entry_price,
                            'position_size': abs(position_amt) * entry_price,
                        }
                        logging.info(f"Found ongoing position for {symbol}: {open_positions[symbol]}")
                except Exception as e:
                    logging.error(f"Error processing position data for symbol {symbol}: {e}", exc_info=True)
                    continue

            # If no ongoing positions are found, log and return the empty dictionary
            if not open_positions:
                logging.info("No ongoing positions found.")
                return {}

            # Calculate cumulative unrealized PNL
            cumulative_unrealized_pnl = sum(pos['unrealized_pnl'] for pos in open_positions.values())
            logging.info(f"Ongoing positions fetched successfully. Cumulative unrealized PNL: {cumulative_unrealized_pnl}")

            return open_positions

        except Exception as e:
            logging.error(f"Unexpected error in check_ongoing_positions: {e}", exc_info=True)
            return {}




    @classmethod
    async def close_client(cls):
        if cls._client_instance:
            await cls._client_instance.close_connection()  # Correct method to close client
            cls._client_instance = None
            print("Client closed successfully.")

    
    async def get_unrealized_pnl(self, symbol: str) -> float:
        try:
            # Ensure client is initialized
            if not await self.ensure_client_initialized():
                return 0.0

            # Fetch the current price for the symbol
            ticker = await self.client.futures_ticker(symbol=symbol) # type: ignore
            current_price = float(ticker['lastPrice'])

            # Fetch all open positions
            positions_info = await self.client.futures_position_information() # type: ignore

            # Find the specific position for the given symbol
            for position in positions_info:
                if position['symbol'] == symbol:
                    entry_price = float(position['entryPrice'])
                    position_amt = float(position['positionAmt'])

                    if position_amt > 0:
                        unrealized_pnl = (current_price - entry_price) * position_amt
                    else:
                        unrealized_pnl = (entry_price - current_price) * abs(position_amt)
                    
                    return unrealized_pnl

            # Return 0 if no position found for the symbol
            return 0.0
        except Exception as e:
            logging.error(f"Error calculating unrealized PNL for {symbol}: {e}")
            await asyncio.sleep(20)
            return 0.0

    
    
    async def fetch_maintenance_margin_coefficient(self) -> Optional[float]:
        try:
            # Ensure client is initialized
            if not await self.ensure_client_initialized():
                return None
            
            # Fetch account information
            account_info = await self.client.futures_account() # type: ignore

            # Extract maintenance margin info from account settings
            if 'totalInitialMargin' in account_info and 'totalMaintenanceMargin' in account_info:
                initial_margin = float(account_info['totalInitialMargin'])
                maintenance_margin = float(account_info['totalMaintenanceMargin'])

                # Avoid division by zero
                if initial_margin > 0:
                    return maintenance_margin / initial_margin
                else:
                    logging.warning("Initial margin is zero. Cannot calculate maintenance margin coefficient.")
                    return None
            else:
                logging.warning("Failed to fetch maintenance margin coefficient from account information.")
                return None

        except Exception as e:
            logging.error(f"Error fetching maintenance margin coefficient: {e}")
            await asyncio.sleep(20)
            return None


    async def calculate_available_margin(self, open_positions) -> Optional[float]:
        try:
            account_info = await self.client.futures_account_balance() # type: ignore

            # Fetch the current BNFCR balance
            bnfcr_balance = float(next((asset['balance'] for asset in account_info['assets'] if asset['asset'] == 'BNFCR'), 0.0))
            logging.info(f"Current BNFCR balance: {bnfcr_balance}")

            used_margin = 0.0

            for symbol, position_details in open_positions.items():
                position_size = position_details['position_size']
                entry_price = position_details['entry_price']
                leverage = position_details.get('leverage', 1)  # Fetch leverage from position details if available

                # Fetch maintenance margin coefficient dynamically
                maintenance_margin_coefficient = await self.fetch_maintenance_margin_coefficient()
                if maintenance_margin_coefficient is None:
                    logging.warning(f"Failed to fetch maintenance margin for {symbol}. Skipping position.")
                    continue

                # Calculate used margin for the position
                used_margin += position_size * entry_price * maintenance_margin_coefficient / leverage

            # Calculate the total margin used and the remaining available margin
            total_margin_used = used_margin

            # Calculate the available margin considering current balance and used margin
            available_margin = bnfcr_balance - total_margin_used

            logging.info(f"Available margin: {available_margin}")
            return available_margin

        except Exception as e:
            logging.error(f"Error calculating available margin: {str(e)}")
            await asyncio.sleep(20)
            return None



    def extract_signals(self, results, ma_signal, tema_signal):
        ma_signals, tema_signals = {}, {}
        index = 0
        if ma_signal:
            ma_signals['1m'], ma_signals['3m'], ma_signals['5m'] = results[index:index + 3]
            index += 3
        if tema_signal:
            tema_signals['1m'], tema_signals['3m'], tema_signals['5m'] = results[index:index + 3]
        return ma_signals, tema_signals


    def extract_technical_indicators(self, df_1m):
        close_1m = df_1m.iloc[-1]['close']
        ma_7_1m = df_1m.iloc[-1]['MA_7']
        ma_14_1m = df_1m.iloc[-1]['MA_14']
        ma_30_1m = df_1m.iloc[-1]['MA_30']
        tema_9m_1m = df_1m.iloc[-1]['TEMA']
        tema_30m_1m = df_1m.iloc[-30]['TEMA'] if len(df_1m) > 30 else None
        return close_1m, ma_7_1m, ma_14_1m, ma_30_1m, tema_9m_1m, tema_30m_1m


    def conditions_met(self, ma_signals, tema_signals, close_1m, ma_7_1m, ma_14_1m, ma_30_1m, tema_9m_1m):
        ma_condition_met = all(signal for signal in ma_signals.values())
        tema_condition_met = all(signal for signal in tema_signals.values())
        logging.debug(f"MA Condition Met: {ma_condition_met}, TEMA Condition Met: {tema_condition_met}")

        if ma_condition_met and tema_condition_met:
            logging.info(f"All conditions met for opening short position.")
            logging.debug(f"Position Details: Close: {close_1m}, MA_7: {ma_7_1m}, TEMA_9: {tema_9m_1m}")
            return True

        if tema_9m_1m < ma_7_1m and tema_9m_1m < ma_14_1m and tema_9m_1m < ma_30_1m:
            logging.info(f"TEMA crossed below moving averages, considering short position.")
            return True

        return False