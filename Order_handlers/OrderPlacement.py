# Remove unnecessary imports that cause circular dependencies
import os
import logging
from binance.client import AsyncClient
from binance import exceptions
from aiohttp import ClientError
from binance.exceptions import BinanceAPIException, BinanceRequestException, BinanceOrderException, BinanceOrderUnknownSymbolException
from decimal import Decimal, ROUND_UP, ROUND_DOWN
import asyncio
import pandas as pd
import traceback
from dotenv import load_dotenv
import random
import json
import dotenv
import hmac
import hashlib
import aiohttp
from typing import Optional, Dict, List
import uuid
from colorama import Fore, Style, Back
import datetime, time
from data.AccountManager.AccountManager import AccountManager  # Example of necessary import
from data.StopLossOrders import StopLossOrders
from data.OrderManagerTakeProfit import OrderManagerTakeProfit  # Example of necessary import
# from data.BinanceManagerHandler import BinanceShortManager  # Remove this import to break the circular dependency
from logger_config import get_logger

logging.basicConfig(level=logging.INFO)
dotenv.load_dotenv()  # Load environment variables from a .env file
# Set the event loop policy to SelectorEventLoop for compatibility with aiodns
if os.name == 'nt':  # Check if the OS is Windows
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Get the logger for this module
logger = get_logger(__name__, blue_background=True)

# Initialize the Binance client with appropriate API keys
api_key = os.getenv('BINANCE_API_KEY')
api_secret = os.getenv('BINANCE_API_SECRET')
client = AsyncClient.create(api_key=api_key, api_secret=api_secret)


# Take_profit_orders_handler = OrderManagerTakeProfit

# Constant for retrying to place the SHORT ORDER correctly upon attempts or over transient NETWORK Errors.
MAX_RETRIES = 3
# Define the base directory for saving order files
# BASE_DIR = r"T:\Trading BOT Crypto - Profit\Profit"



class OrderPlacement:
    def __init__(self, client, symbol):
        self.session = None
        self.client = client
        self.symbols = symbol
        self.ma_windows = [5, 10, 20]  # Example moving average windows
        self.api_key = os.getenv('BINANCE_API_KEY')
        self.api_secret = os.getenv('BINANCE_API_SECRET')

        if not self.client:
            raise ValueError("Binance client is not initialized. API key and secret are required.")

        self.running_positions = {}
        self.open_orders = {}
        self.checking_balance_for_placing_order = AccountManager(client, api_key, api_secret)
        self.order_manager = OrderManagerTakeProfit(client, symbol)
        self.stop_loss_manager = StopLossOrders(client, symbol=symbol)
        self.trade_records = []
        self.tax_report_file = "tax_report.xlsx"
        logger.info("OrderPlacement class initialized successfully")

    async def initialize_client(self, api_key, api_secret):
        """ Initialize Binance async client """
        try:
            self.client = await AsyncClient.create(api_key=api_key, api_secret=api_secret)
            self.order_manager.client = self.client  # Ensure order manager has access to the client
            self.stop_loss_manager.client = self.client  # Ensure stop loss manager has access to the client
            # self.account_manager.client = self.client
            logging.info("Binance async client initialized")
        except Exception as e:
            logging.error(f"Error initializing Binance async client: {e}")
            await asyncio.sleep(20)


    async def initialize_session(self):
        if not self.session:
            self.session = aiohttp.ClientSession()  # Create session once

    # async def fetch_positions(self):
    #     try:
    #         positions = await self.client.futures_position_information()
    #         return positions
    #     except Exception as e:
    #         logging.error(f"Error fetching positions: {e}")
    #         await asyncio.sleep(20)
    #         return []
    async def fetch_current_prices(self, symbols):
        prices = {}
        try:
            await self.initialize_session()
            if not self.session:
                logging.error("Session not initialized.")
                return prices

            logging.info(f"Fetching prices for symbols: {symbols}")

            for symbol in symbols:
                if not isinstance(symbol, str) or len(symbol) < 3:  # Basic validation for symbol length
                    logging.error(f"Invalid symbol format: {symbol}")
                    prices[symbol] = None
                    continue

                logging.info(f"Fetching price for symbol: {symbol}")
                for attempt in range(5):  # Retry up to 5 times
                    async with self.session.get('https://api.binance.com/api/v3/ticker/price', params={'symbol': symbol}) as response:
                        if response.status == 200:
                            data = await response.json()
                            logging.info(f"[{symbol}] API response: {data}")

                            # Directly access the price for the symbol
                            price = float(data['price'])
                            prices[symbol] = price if price > 0 else None
                            break  # Exit retry loop on success
                        else:
                            logging.error(f"Failed to get a valid response from ticker endpoint for {symbol}. Status: {response.status}")
                            prices[symbol] = None
                            await asyncio.sleep(2)  # Wait before retrying
                else:
                    logging.error(f"[{symbol}] All attempts to fetch price failed.")
        except aiohttp.ClientError as e:
            logging.error(f"Network or client error fetching prices: {str(e)}", exc_info=True)
        except ValueError as e:
            logging.error(f"Value error while processing prices: {str(e)}", exc_info=True)
        except Exception as e:
            logging.error(f"Unexpected error fetching prices: {str(e)}", exc_info=True)

        return prices

    async def calculate_position_size(self, symbol: List[str]) -> Decimal:
        try:
            symbol_info = await self.get_symbol_info(symbol)

            if not symbol_info:
                raise ValueError("Failed to fetch symbol information.")
            
            quantity_precision = symbol_info.get('quantity_precision', 0)
            min_trade_amount = symbol_info.get('min_trade_amount', Decimal('0'))

            available_margin = await self.fetch_futures_balance('BNFCR')
            if available_margin is None:
                raise ValueError("Failed to fetch available margin.")
            
            available_margin = Decimal(available_margin)

            margin_to_use = available_margin * Decimal('0.20')
            current_prices = await self.fetch_current_prices(symbol)

            current_price = Decimal(current_prices.get(symbol[0], 0.0))  # Access and convert to Decimal
            if current_price <= 0:
                raise ValueError("Invalid current price fetched.")

            max_leverage = await self.get_symbol_max_leverage(symbol)
            leverage = min(max(35, max_leverage), 125)

            position_size = (margin_to_use * leverage) / current_price
            min_notional_value = Decimal('5')
            min_quantity = min_notional_value / current_price

            if position_size < min_quantity:
                logging.warning(f"Insufficient funds. Position size ({position_size}) is less than minimum quantity ({min_quantity}).")
                raise RuntimeError("Insufficient funds for position size. Please deposit funds or transfer assets to the futures wallet.")

            if quantity_precision > 0:
                position_size = position_size.quantize(Decimal(10) ** -quantity_precision, rounding=ROUND_DOWN)

            if position_size < min_trade_amount:
                position_size = min_trade_amount
            
            return position_size

        except Exception as e:
            logging.error(f"Error in calculating position size: {e}", exc_info=True)
            raise RuntimeError(f"Error in calculating position size: {e}")

    async def get_symbol_max_leverage(self, symbol: List[str]) -> int:
        try:
            # Fetch leverage information from the Binance API
            exchange_info = await self.client.futures_exchange_info()

            for s in exchange_info['symbols']:
                if s['symbol'] == symbol:
                    leverage_filter = s.get('leverageFilter')
                    if leverage_filter:
                        max_leverage = int(leverage_filter['maxLeverage'])
                        return min(max_leverage, 125)  # You can adjust this as needed

            # Log an error if the symbol is not found
            logging.error(f"Symbol {symbol} not found in exchange info.")
            return 20  # Default leverage if symbol is not found

        except AttributeError as e:
            logging.error(f"AttributeError: {e}")
            return 20  # Default leverage in case of error

        except Exception as e:
            logging.error(f"Error fetching maximum leverage for {symbol}: {e}")
            return 20  # Default leverage in case of error

    async def apply_leverage(self, symbol: List[str]) -> Optional[Dict]:
        """
        Apply the maximum allowable leverage to the specified symbol.
        """
        try:
            # Fetch maximum leverage allowed for the symbol
            max_leverage = await self.get_symbol_max_leverage(symbol)
            
            # Apply leverage (e.g., reduce by 1 or use a specific factor for safety)
            leverage = min(max(25, max_leverage), 125)  # Safety margin and upper limit
            
            logging.info(f"Applying leverage of {leverage} to symbol {symbol}.")
            
            # Apply the leverage value
            response = await self.client.futures_change_leverage(symbol=symbol, leverage=leverage)
            return response  # Return response for verification

        except Exception as e:
            logging.error(f"Error applying leverage for {symbol}: {e}", exc_info=True)
            return None

    async def get_symbol_info(self, symbol: List[str]):
        """
        Fetch detailed symbol information including maximum leverage and return it.
        """
        try:
            # Ensure client is properly initialized
            if not hasattr(self.client, 'futures_exchange_info'):
                raise AttributeError("Client does not have the method 'futures_exchange_info'")

            # Fetch symbol information from the Binance API
            exchange_info = await self.client.futures_exchange_info()

            # Find the symbol info
            symbol_info = next((s for s in exchange_info['symbols'] if s['symbol'] == symbol), None)
            
            if symbol_info:
                # Extract relevant information
                max_leverage = float(symbol_info.get('maxLeverage', 1.0))
                min_trade_amount = Decimal(symbol_info.get('filters')[1].get('minQty', '0'))
                min_price_movement = Decimal(symbol_info.get('filters')[0].get('tickSize', '0'))
                price_precision = int(symbol_info.get('quotePrecision', 0))
                quantity_precision = int(symbol_info.get('baseAssetPrecision', 0))

                # Return a dictionary with symbol information
                return {
                    'max_leverage': max_leverage,
                    'min_trade_amount': min_trade_amount,
                    'min_price_movement': min_price_movement,
                    'price_precision': price_precision,
                    'quantity_precision': quantity_precision
                }
            else:
                logging.error(f"Symbol {symbol} not found in exchange info.")
                return None

        except AttributeError as e:
            logging.error(f"AttributeError: {e}")
            return None

        except Exception as e:
            logging.error(f"Error fetching symbol info for {symbol}: {e}")
            return None

    @staticmethod
    def generate_signature(params: dict, api_secret: str) -> str:
        query_string = '&'.join([f"{key}={value}" for key, value in sorted(params.items())])
        signature = hmac.new(api_secret.encode(), query_string.encode(), hashlib.sha256).hexdigest()
        return signature

    @staticmethod
    async def fetch_futures_balance(asset: str) -> Decimal:
        api_key = os.getenv('BINANCE_API_KEY')
        api_secret = os.getenv('BINANCE_API_SECRET')

        if not api_key or not api_secret:
            logging.error("API key or secret not found in environment variables.")
            return Decimal('0')

        url = "https://fapi.binance.com/fapi/v2/balance"
        headers = {'X-MBX-APIKEY': api_key}
        timestamp = int(time.time() * 1000)
        params = {'timestamp': str(timestamp)}
        params['signature'] = OrderPlacement.generate_signature(params, api_secret)

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, params=params) as response:
                    if response is None:
                        logging.error("Failed to get a valid response.")
                        return Decimal('0')
                    
                    if response.status == 200:
                        balance_data = await response.json()
                        for asset_data in balance_data:
                            if asset_data['asset'] == asset:
                                return Decimal(asset_data['balance'])
                    else:
                        logging.error(f"Failed to fetch balance from Binance. Status code: {response.status}")
        except Exception as e:
            logging.error(f"Error fetching balance: {str(e)}", exc_info=True)

        return Decimal('0')
        
    @staticmethod
    async def fetch_klines(client, symbol: List[str], interval='1m', start_time='2024-08-01', end_time=None):
        """
        Fetch klines for a given symbol without retries and additional logic.
        :param client: Binance API client instance
        :param symbol: Trading pair symbol, e.g., 'BTCUSDT'
        :param interval: Kline interval, e.g., '1m', '5m', '1h'
        :param start_time: Start time for fetching klines
        :param end_time: End time for fetching klines (optional)
        :return: DataFrame of fetched klines, or None if an error occurs
        """
        try:
            logging.info(f"Fetching klines for {symbol} with interval {interval}, Start time: {start_time}, End time: {end_time}")
            
            # Fetch klines from the Binance client
            klines = await client.futures_klines(
                symbol=symbol,
                interval=interval,
                start_str=start_time,
                end_str=end_time,
                limit=500
            )

            # Return klines if available
            if klines:
                return klines
            else:
                logging.warning(f"No data returned for symbol {symbol}.")
                await asyncio.sleep(10)

                return None

        except (BinanceRequestException, asyncio.TimeoutError) as e:
            logging.error(f"Network or API error while fetching klines for {symbol}: {e}")
            return None

        except BinanceAPIException as e:
            logging.error(f"Binance API exception for symbol {symbol}: {e}")
            return None

        except Exception as e:
            logging.error(f"Unexpected error while fetching klines for {symbol}: {e}")
            return None
    
    async def get_quantity_limits(self, symbol: List[str]):
        try:
            exchange_info = await self.client.futures_exchange_info()
            symbol_info = next((s for s in exchange_info['symbols'] if s['symbol'] == symbol), None)
            
            if not symbol_info:
                raise ValueError(f"Symbol {symbol} not found in exchange info.")
            
            filters = symbol_info.get('filters', [])
            lot_size_filter = next((f for f in filters if f['filterType'] == 'LOT_SIZE'), None)
            
            if not lot_size_filter:
                raise ValueError(f"LOT_SIZE filter not found for {symbol}.")
            
            min_qty = Decimal(lot_size_filter['minQty'])
            max_qty = Decimal(lot_size_filter['maxQty'])
            step_size = Decimal(lot_size_filter['stepSize'])
            
            return min_qty, max_qty, step_size
        
        except Exception as e:
            logging.error(f"Error fetching quantity limits for {symbol}: {str(e)}")
            return Decimal('0'), Decimal('0'), Decimal('0')

    async def fetch_running_positions(self):
        if self.client is None:
            logging.error("Binance client is not initialized. Cannot fetch running positions.")
            return {}

        retry_count = 0
        max_retries = 3
        while retry_count < max_retries:
            try:
                positions = await self.client.futures_account()
                logging.debug(f"Fetched positions data: {positions}")

                if 'positions' in positions and positions['positions'] is not None:
                    self.running_positions = {
                        position['symbol']: {
                            'positionAmt': Decimal(position['positionAmt']) if position['positionAmt'] else Decimal('0'),
                            'entryPrice': Decimal(position['entryPrice']) if position['entryPrice'] else Decimal('0')
                        }
                        for position in positions['positions']
                        if float(position['positionAmt']) != 0
                    }
                    logging.info(f"Running positions fetched and updated: {self.running_positions}")
                    return self.running_positions
                else:
                    logging.warning("No 'positions' key in the fetched data or 'positions' is None.")
                    self.running_positions = {}
                    return self.running_positions

            except RuntimeError as e:
                if "Session is closed" in str(e):
                    logging.error("Session is closed. Attempting to reinitialize the client...")
                    await self.initialize_client(api_key, api_secret)
                else:
                    logging.error(f"Error fetching running positions: {str(e)}", exc_info=True)
                await asyncio.sleep(2)
                retry_count += 1
            except Exception as e:
                logging.error(f"Unexpected error fetching running positions: {str(e)}", exc_info=True)
                break

        logging.error("Max retries exceeded while fetching running positions.")
        self.running_positions = {}
        return self.running_positions

    async def check_ongoing_positions(self, symbol):
        running_positions = await self.fetch_running_positions()
        if symbol in running_positions and running_positions[symbol]['positionAmt'] != 0:
            return True
        return False

    async def check_ongoing_short_position(self, symbol):
        try:
            # Fetch and update running positions
            await self.fetch_running_positions()

            # Ensure running_positions is a dictionary
            if not isinstance(self.running_positions, dict):
                logging.error("Running positions data is not in the expected format.")
                return {}

            # Identify ongoing short positions
            ongoing_short_positions = {
                symbol: position
                for symbol, position in self.running_positions.items()
                if position.get('positionSide') == 'SHORT' and float(position.get('positionAmt', 0)) > 0
            }

            # Log results
            if ongoing_short_positions:
                logging.info(f"Found ongoing short positions: {ongoing_short_positions}")
            else:
                logging.info("No ongoing short positions found.")

            return ongoing_short_positions

        except Exception as e:
            logging.error(f"Error checking ongoing short positions: {e}", exc_info=True)
            await asyncio.sleep(20)
            return {}

    async def calculate_used_margin(self) -> tuple[Decimal, Decimal]:
        try:
            # Fetch the futures wallet balance for BNFCR
            balance_info = await self.fetch_futures_balance('BNFCR')
            if balance_info is None or not isinstance(balance_info, dict):
                logging.error("Failed to fetch or parse BNFCR futures balance.")
                return Decimal('0.0'), Decimal('0.0')  # Return default values

            # Get the total wallet balance for USDⓈ-M Futures
            total_balance_str = balance_info.get('totalWalletBalance', '0')
            total_balance = Decimal(total_balance_str)

            # Fetch open positions in USDⓈ-M futures
            open_positions = await self.client.futures_position_information()
            if open_positions is None or not isinstance(open_positions, list):
                logging.error("Failed to fetch or parse open positions.")
                return Decimal('0.0'), Decimal('0.0')  # Return default values

            # Initialize total margin used
            total_margin_used = Decimal('0.0')

            # Loop through each position to calculate margin used
            for position in open_positions:
                symbol = position.get('symbol')
                if not symbol:
                    continue

                position_size_str = position.get('positionAmt', '0')
                entry_price_str = position.get('entryPrice', '0')
                leverage_str = position.get('leverage', '1')

                # Convert string values to Decimal for accuracy
                position_size = Decimal(position_size_str)
                entry_price = Decimal(entry_price_str)
                leverage = Decimal(leverage_str)

                # Ensure valid position size and entry price for margin calculation
                if position_size != 0 and entry_price != 0:
                    # Calculate margin used for this position (abs for both short and long handling)
                    position_margin = (abs(position_size) * entry_price) / leverage
                    total_margin_used += position_margin

            # Calculate available margin by subtracting used margin from total balance
            available_margin = total_balance - total_margin_used

            # Calculate used margin as a percentage of total balance
            used_margin_percentage = (total_margin_used / total_balance) * 100 if total_balance > 0 else Decimal('0.0')

            # Detailed log for margin calculation
            logging.info(f"Total balance: {total_balance}, Total margin used: {total_margin_used}, "
                        f"Available margin: {available_margin}, Used margin percentage: {used_margin_percentage}%")

            return available_margin, used_margin_percentage

        except Exception as e:
            logging.error(f"Error calculating used margin: {str(e)}", exc_info=True)
            return Decimal('0.0'), Decimal('0.0')  # Return default values on error

    async def calculate_margin_required(self, symbol: str, entry_price: Decimal, quantity: Decimal) -> Decimal:
        try:
            # Fetch symbol info to get maintenance margin rate
            exchange_info = await self.client.get_exchange_info()
            symbol_info = next((item for item in exchange_info['symbols'] if item['symbol'] == symbol), None)

            if not symbol_info:
                raise ValueError(f"Symbol information not found for {symbol}")

            # Fetch maintenance margin rate, use default if not found
            maintenance_margin_rate = Decimal(symbol_info.get('maintenanceMarginRate', '0.005'))  # Example default value

            # Calculate notional value of the position
            notional_value = entry_price * quantity

            # Calculate margin required
            margin_required = notional_value * maintenance_margin_rate

            return margin_required
        except KeyError as e:
            logging.error(f"KeyError in calculate_margin_required: {e}", exc_info=True)
            raise
        except Exception as e:
            logging.error(f"Unexpected error in calculate_margin_required: {e}", exc_info=True)
            raise

    async def risk_management_check(self, symbol: str) -> bool:
        try:
            ongoing_short_positions = await self.check_ongoing_short_position(symbol=symbol)

            if not ongoing_short_positions:
                logging.info("No ongoing short positions to evaluate for risk management.")
                return True

            total_maintenance_margin = Decimal('0')
            for position in ongoing_short_positions.values():
                maint_margin = position.get('maintMargin')
                if maint_margin is not None:
                    total_maintenance_margin += Decimal(maint_margin)
                else:
                    logging.warning(f"Position missing 'maintMargin': {position}")

            # Logging the total maintenance margin for better tracking
            logging.info(f"Total maintenance margin: {total_maintenance_margin}")

            if total_maintenance_margin >= Decimal('0.75'):
                logging.warning(f"Maintenance margin is above 75% ({total_maintenance_margin}). Proceed with manual review.")
                return False
            elif total_maintenance_margin >= Decimal('0.70'):
                logging.info(f"Maintenance margin is above 70% ({total_maintenance_margin}). Proceed with caution.")
                return True
            elif total_maintenance_margin < Decimal('0.35'):
                logging.info(f"Maintenance margin is below 35% ({total_maintenance_margin}). Safe to open new positions.")
                return True
            else:
                logging.info(f"Maintenance margin is between 35% and 70% ({total_maintenance_margin}). Monitor closely.")
                return True

        except Exception as e:
            logging.error(f"Error during risk management check for symbol {symbol}: {str(e)}", exc_info=True)
            return False


    async def open_short_position(self, symbols, entry_price: Optional[Decimal]) -> Optional[Dict]:
        for symbol in symbols:
            try:
                # Fetch the current price
                current_price_float = await self.fetch_current_prices(symbol)
                if isinstance(current_price_float, (float, int)):
                    if current_price_float <= 0:
                        logging.error(f"[{symbol}] Current price is not valid. Aborting short position.")
                        return None
                else:
                    logging.error(f"[{symbol}] Current price fetched is not a valid number: {current_price_float}. Aborting short position.")
                    return None

                # Convert current price to Decimal
                current_price = Decimal(current_price_float)

                # Determine the precision based on current_price
                def get_precision(price: Decimal) -> Decimal:
                    exponent = price.as_tuple().exponent
                    if isinstance(exponent, int):
                        return Decimal(-exponent)  # Return as Decimal
                    else:
                        raise TypeError(f"Unexpected exponent type: {type(exponent)}")

                # Get precision as a Decimal
                precision = get_precision(current_price)

                # Use the fetched price if entry_price is not provided
                if entry_price is None:
                    entry_price = current_price
                else:
                    # Create a Decimal for quantization with the correct precision
                    quantize_format = Decimal('1.' + '0' * int(precision))  # Convert to int for quantization
                    entry_price = entry_price.quantize(quantize_format, rounding=ROUND_DOWN)

                logging.info(f"[{symbol}] Performing pre-order risk management check.")
                ongoing_short_positions = await self.check_ongoing_short_position(symbol)

                if ongoing_short_positions:
                    total_maintenance_margin = await self.fetch_futures_balance(asset="BNFCR")
                    logger.info("Current Wallet Balance:")
                    for position in ongoing_short_positions.values():
                        maint_margin = position.get('maintMargin')
                        if maint_margin:
                            total_maintenance_margin += Decimal(maint_margin)
                        else:
                            logging.warning(f"[{symbol}] Position missing 'maintMargin': {position}")

                    if total_maintenance_margin >= Decimal('7.5'):
                        logging.warning(f"[{symbol}] Total maintenance margin is above 75% ({total_maintenance_margin}%). Manual review needed. Aborting order placement.")
                        return None

                # Calculate the position size
                available_amount = await self.calculate_position_size(symbols)

                # Fetch symbol precision and leverage
                exchange_info = await self.client.get_exchange_info()
                symbol_info = next((item for item in exchange_info['symbols'] if item['symbol'] == symbol), None)
                if not symbol_info:
                    logging.error(f"Symbol information not found for {symbol}.")
                    raise ValueError(f"Symbol information not found for {symbol}")

                price_precision = symbol_info['quotePrecision']
                quantity_precision = symbol_info['baseAssetPrecision']
                entry_price = entry_price.quantize(Decimal('1.' + '0' * price_precision))

                # Subtract 50% from the available amount for risk management
                adjusted_amount = available_amount * Decimal('0.3')

                # Fetch and apply leverage
                max_leverage = await self.get_symbol_max_leverage(symbols)
                leverage = min(max(30, int(max_leverage)), 125)
                logging.info(f"[{symbol}] Applying leverage: {leverage}")
                if not await self.apply_leverage(symbols):
                    logging.error(f"[{symbol}] Failed to apply leverage.")
                    await asyncio.sleep(20)
                    return None

                # Calculate the initial quantity based on the adjusted amount and leverage
                quantity = (adjusted_amount * leverage / entry_price).quantize(Decimal('1.' + '0' * quantity_precision * price_precision))
                logging.info(f"[{symbol}] Calculated position size (quantity): {quantity}")

                # Place the short order
                logging.info(f"[{symbol}] Placing short order with quantity: {quantity} at entry price: {entry_price}")
                for attempt in range(MAX_RETRIES):
                    try:
                        response = await self.client.futures_create_order(
                            symbol=symbol,
                            side='SELL',
                            positionSide='SHORT',
                            type='MARKET',
                            leverage=leverage,
                            quantity=quantity,
                            newClientOrderId=uuid.uuid4().hex
                        )
                        if 'orderId' in response:
                            order_id = response['orderId']
                            logging.info(f"[{symbol}] Short position opened with quantity {quantity}. Order ID: {order_id}")

                            # Save order details to file
                            await self.save_order_details(symbols, order_id, quantity, entry_price)

                            # **Stop Loss Placement**
                            try:
                                stop_loss_price = await self.stop_loss_manager.calculate_dynamic_stop_loss_short(
                                    symbol=[symbol],
                                    order_id=order_id,
                                    risk_level=Decimal('0.02'),
                                    max_price_diff=Decimal('0.1')
                                )

                                logging.info(f"[{symbol}] Calculated stop loss price: {stop_loss_price}")

                                if stop_loss_price is not None:
                                    stop_loss_price = Decimal(str(stop_loss_price)).quantize(Decimal('1.' + '0' * price_precision))
                                    logging.info(f"[{symbol}] Stop loss price after quantization: {stop_loss_price}")

                                    stop_loss_order_request = {
                                        'symbol': symbol,
                                        'side': 'BUY',
                                        'quantity': str(quantity),
                                        'type': 'STOP_MARKET',
                                        'stopPrice': str(stop_loss_price),
                                        'reduceOnly': True,
                                        'closePosition': True,
                                        'positionSide': 'SHORT',
                                        'timeInForce': 'IMMEDIATE_OR_CANCEL'
                                    }

                                    stop_loss_order_response = await self.client.futures_create_order(**stop_loss_order_request)
                                    logging.info(f"[{symbol}] Stop loss order response: {stop_loss_order_response}")

                            except Exception as e:
                                logging.error(f"[{symbol}] Error in placing stop loss order: {e}", exc_info=True)
                                return None  # Exit if stop-loss order placement fails

                            # **Take Profit Placement**
                            try:
                                take_profit_price = await self.order_manager.calculate_dynamic_take_profit_short(symbol)
                                logging.info(f"[{symbol}] Calculated take profit price: {take_profit_price}")

                                if take_profit_price is not None:
                                    take_profit_price = Decimal(str(take_profit_price)).quantize(Decimal('1.' + '0' * price_precision))
                                    logging.info(f"[{symbol}] Take profit price after quantization: {take_profit_price}")
                                    # Ensure all required arguments are passed
                                    await self.order_manager.place_take_profit_order_wrapper(symbol, client, entry_price=entry_price, initial_quantity=quantity)
                                    take_profit_order_request = {
                                        'symbol': symbol,
                                        'side': 'BUY',
                                        'quantity': str(quantity),
                                        'type': 'LIMIT',
                                        'price': str(take_profit_price),
                                        'reduceOnly': True,
                                        'closePosition': True,
                                        'positionSide': 'SHORT',
                                        'timeInForce': 'GTC'
                                    }

                                    take_profit_order_response = await self.client.futures_create_order(**take_profit_order_request)
                                    logging.info(f"[{symbol}] Take profit order response: {take_profit_order_response}")

                            except Exception as e:
                                logging.error(f"[{symbol}] Error in placing take profit order: {e}", exc_info=True)

                            return response
                        else:
                            logging.warning(f"[{symbol}] Short position order response did not contain 'orderId'. Response: {response}")

                    except BinanceAPIException as e:
                        logging.error(f"[{symbol}] Binance API Exception: {e}", exc_info=True)
                    except Exception as e:
                        logging.error(f"[{symbol}] Unexpected error: {e}", exc_info=True)

                    # Retry with exponential backoff
                    await asyncio.sleep(2 ** attempt)

                logging.error(f"[{symbol}] Failed to open short position after {MAX_RETRIES} attempts.")
                return None

            except Exception as e:
                logging.error(f"[{symbol}] Error in opening short position: {e}", exc_info=True)
                return None
            finally:
                # Ensure cleanup or logging that should always happen
                logging.info(f"[{symbol}] Finished attempting to open short position. Performing any necessary final cleanup or state reset.")

    # async def open_long_position(
    #     self,
    #     symbol: str,
    #     entry_price: Decimal,
    #     position_size: Decimal = Decimal('5.0')
    # ) -> dict | None:
    #     try:
    #         logging.info(f"[{symbol}] Type of position_size: {type(position_size)}")

    #         # Ensure position_size is a Decimal
    #         if not isinstance(position_size, Decimal):
    #             logging.error(f"[{symbol}] Invalid position_size: {position_size}. It should be a Decimal.")
    #             await asyncio.sleep(20)
    #             return None

    #         # Fetch precision levels for the symbol
    #         price_precision, quantity_precision = await self.stop_loss_manager.get_symbol_precision(symbol)
    #         if price_precision is None or quantity_precision is None:
    #             logging.error(f"[{symbol}] Failed to fetch precision for {symbol}.")
    #             await asyncio.sleep(20)
    #             return None

    #         # Fetch leverage and ensure it's a valid Decimal
    #         try:
    #             leverage = Decimal(await self.get_symbol_max_leverage(symbol))
    #             if leverage <= Decimal('0'):
    #                 logging.error(f"[{symbol}] Invalid leverage value: {leverage}. Leverage should be greater than 0.")
    #                 await asyncio.sleep(20)
    #                 return None
    #         except Exception as e:
    #             logging.error(f"[{symbol}] Error fetching leverage for {symbol}. Error: {str(e)}", exc_info=True)
    #             await asyncio.sleep(20)
    #             return None

    #         # Fetch BNFCR balance
    #         try:
    #             balance_result = await OrderPlacement.fetch_futures_balance('BNFCR')
    #             if balance_result is None:
    #                 logging.error(f"[{symbol}] Failed to fetch BNFCR balance. Long Condition checking balance failed.", exc_info=True)
    #                 await asyncio.sleep(20)
    #                 return None
                
    #             # Convert balance result to Decimal
    #             BNFCR_balance = Decimal(balance_result)
    #             if BNFCR_balance < Decimal('0.1'):
    #                 logging.error(f"[{symbol}] Insufficient BNFCR balance ({BNFCR_balance} BNFCR). Minimum required is 0.5 BNFCR.")
    #                 await asyncio.sleep(20)
    #                 return None
    #         except Exception as e:
    #             logging.error(f"[{symbol}] Error fetching or converting BNFCR balance: {str(e)}", exc_info=True)
    #             await asyncio.sleep(20)
    #             return None

    #         # Calculate position size based on balance portion and leverage
    #         balance_portion = (BNFCR_balance * Decimal('0.75')) * Decimal(leverage)
    #         total_position_size = balance_portion.quantize(Decimal(f'0.{"0" * quantity_precision}'))
    #         logging.info(f"[{symbol}] Calculated position size: {total_position_size} BNFCR")

    #         # Ensure Binance client is initialized
    #         if not self.client:
    #             logging.error(f"[{symbol}] Binance client not initialized before placing long order.", exc_info=True)
    #             await self.initialize_client(api_key, api_secret)
    #             if not self.client:
    #                 logging.error(f"[{symbol}] Binance client re-initialization failed.", exc_info=True)
    #                 await asyncio.sleep(20)
    #                 return None

    #         # Check if the balance is sufficient
    #         if BNFCR_balance < Decimal('0.30'):
    #             logging.error(f"[{symbol}] Insufficient BNFCR balance ({BNFCR_balance} BNFCR). Required: {total_position_size} BNFCR.")
    #             await asyncio.sleep(20)
    #             return None

    #         # Quantize entry price and quantity based on precision
    #         entry_price = entry_price.quantize(
    #             Decimal(f'0.{"0" * price_precision}'), rounding=ROUND_UP
    #         )

    #         # Calculate the quantity
    #         quantity = total_position_size.quantize(
    #             Decimal(f'0.{"0" * quantity_precision}'), rounding=ROUND_UP
    #         )

    #         if quantity <= Decimal('0'):
    #             logging.error(f"[{symbol}] Calculated quantity is zero after rounding. Skipping order.")
    #             await asyncio.sleep(20)
    #             return None

    #         logging.info(f"[{symbol}] Placing long order with quantity: {quantity}")
    #         new_order_id_long = uuid.uuid4().hex
            
    #         # Place the market order for long position
    #         long_order = await self.client.futures_create_order(
    #             symbol=symbol,
    #             side='BUY',
    #             type='MARKET',
    #             quantity=str(quantity),
    #             newClientOrderId=new_order_id_long,
    #             positionSide='LONG',  # Ensure correct position side for long
    #             # reduceOnly=False,  # This is an opening order, so reduceOnly is False
    #             workingType='CONTRACT_PRICE',  # Optional, define working type if needed
    #             selfTradePreventionMode='NONE'  # Adjust based on strategy
    #         )

    #         if not long_order or 'orderId' not in long_order:
    #             logging.error(f"[{symbol}] Failed to place long order.")
    #             await asyncio.sleep(20)
    #             return None

    #         self.open_orders[symbol] = long_order
    #         logging.info(f"[{symbol}] Long position opened with quantity {quantity}. Order ID: {long_order['orderId']}")

    #         # Convert entry_price to float for compatibility
    #         entry_price_float = float(entry_price)

    #         # Place the stop loss order
    #         stop_loss_price = await self.stop_loss_manager.calculate_dynamic_stop_loss_long(symbol, entry_price_float)
    #         if stop_loss_price:
    #             logging.info(f"[{symbol}] Calculated Stop Loss: Price={stop_loss_price}")
    #             stop_loss_result = await self.stop_loss_manager.place_stop_loss_order_long(
    #                 symbol=symbol,
    #                 stop_loss_price=str(stop_loss_price),
    #                 quantity=str(quantity)
    #             )
    #             if stop_loss_result:
    #                 logging.info(f"[{symbol}] Stop loss order placed successfully.")
    #             else:
    #                 logging.error(f"[{symbol}] Failed to place stop loss order.")
    #                 await asyncio.sleep(20)
    #         else:
    #             logging.warning(f"[{symbol}] Stop loss calculation returned None. Skipping Stop Loss order.")
    #             await asyncio.sleep(10)

    #         # Place the take profit order
    #         take_profit_info = await self.order_manager.calculate_dynamic_take_profit_long(symbol, Decimal(entry_price_float))
    #         if take_profit_info:
    #             take_profit_price = Decimal(take_profit_info.get('price', '0')).quantize(Decimal('1.' + '0' * price_precision))
    #             take_profit_quantity = Decimal(take_profit_info.get('quantity', '0')).quantize(Decimal('1.' + '0' * quantity_precision))
    #             logging.info(f"[{symbol}] Calculated Take Profit: Price={take_profit_price}, Quantity={take_profit_quantity}")
    #             take_profit_result = await self.order_manager.place_take_profit_order_wrapper(
    #                 symbol=symbol,
    #                 entry_price=Decimal(entry_price_float),
    #                 initial_quantity=quantity
    #             )
    #             if take_profit_result:
    #                 logging.info(f"[{symbol}] Take profit order placed successfully.")
    #             else:
    #                 logging.error(f"[{symbol}] Failed to place take profit order.")
    #                 await asyncio.sleep(20)
    #         else:
    #             logging.warning(f"[{symbol}] Take profit calculation returned None. Skipping Take Profit order.")
    #             await asyncio.sleep(10)

    #         return long_order

    #     except Exception as e:
    #         logging.error(f"[{symbol}] Error opening long position: {str(e)}", exc_info=True)
    #         await asyncio.sleep(20)
    #         return None


    # async def place_short_order(self, symbol, quantity, current_price, side="SELL"):
    #     try:
    #         logging.info(f"Placing short order: Symbol={symbol}, Side={side}, Quantity={quantity}, Current Price={current_price}")
    #         order_response = await self.client.futures_create_order(
    #             symbol=symbol,
    #             side=side,
    #             type='MARKET',
    #             quantity=quantity,
    #             reduceOnly=False,
    #             priceProtection=True,
    #             newClientOrderId=str(uuid.uuid4())
    #         )
    #         logging.info(f"Order response: {order_response}")

    #         if 'orderId' in order_response:
    #             order_id = order_response['orderId']
    #             logging.info(f"Order ID for {symbol}: {order_id}")

    #             with open(f"{symbol}_order.txt", "w") as f:
    #                 f.write(f"Order ID: {order_id}\n")
    #                 f.write(f"Quantity: {quantity}\n")
    #                 f.write(f"Current Price: {current_price}\n")

    #             # Immediately place stop loss order
    #             stop_loss_price = current_price * 0.995  # Example: 0.5% above entry price
    #             await self.place_short_order(symbol, stop_loss_price, quantity)

    #             return order_response
    #         else:
    #             logging.error(f"No order ID in response for {symbol}: {order_response}")
    #             return None

    #     except Exception as e:
    #         logging.error(f"Error placing short order for {symbol}: {str(e)}", exc_info=True)
    #         return None

    async def save_order_details(self, symbol: List[str], order_id: str, quantity: Decimal, entry_price: Decimal):
        try:
            # Define the path and filename
            directory = f"T:\\Trading BOT Crypto - Profit\\Profit"
            if not os.path.exists(directory):
                os.makedirs(directory)
            
            filename = os.path.join(directory, f"{symbol}_order.txt")

            # Write order details to file
            with open(filename, "a") as f:
                f.write(f"Order ID: {order_id}\n")
                f.write(f"Quantity: {quantity}\n")
                f.write(f"Entry Price: {entry_price}\n")
                f.write(f"Timestamp: {datetime.datetime.now().isoformat()}\n\n")
            
            logging.info(f"[{symbol}] Order details saved to {filename}")
        
        except IOError as e:
            logging.exception(f"[{symbol}] Error writing order details to file: {e}", exc_info=True)

    async def monitor_and_log_orders(self):
        try:
            # Directory where the order details will be saved
            directory = f"T:\\Trading BOT Crypto - Profit\\Profit"
            if not os.path.exists(directory):
                os.makedirs(directory)

            while True:
                for symbol in list(self.open_orders.keys()):
                    try:
                        # Fetch order info
                        order_info = await self.client.futures_get_order(
                            symbol=symbol,
                            orderId=self.open_orders[symbol]['order_id']
                        )

                        if order_info:
                            status = order_info.get('status')
                            if status in ['FILLED', 'CANCELED']:
                                logging.info(f"Order {self.open_orders[symbol]['order_id']} for {symbol} is closed with status {status}")

                                # Remove the order from open orders
                                del self.open_orders[symbol]

                                # Log trade information
                                await self.log_trade(symbol, order_info)

                                # Define the filename for the order details
                                filename = os.path.join(directory, f"{symbol}_order.txt")

                                # Write order details to the file
                                with open(filename, "a") as f:
                                    f.write(f"Order ID: {order_info.get('orderId')}\n")
                                    f.write(f"Status: {status}\n")
                                    unrealized_pnl = await self.calculate_unrealized_pnl(symbol)
                                    f.write(f"Unrealized Profit/Loss: {unrealized_pnl}\n")
                                    f.write(f"Timestamp: {datetime.datetime.now().isoformat()}\n\n")

                                # Append to the general monitoring log
                                monitoring_log_filename = os.path.join(directory, "monitoring_log.txt")
                                with open(monitoring_log_filename, "a") as f:
                                    f.write(f"{symbol} order is closed with status {status}.\n")

                                # Update tax report
                                self.update_tax_report()

                            else:
                                logging.info(f"Order {self.open_orders[symbol]['order_id']} for {symbol} is still open")

                    except Exception as order_error:
                        logging.error(f"Error processing order for {symbol}: {str(order_error)}")

                # Wait before checking the next batch of orders
                await asyncio.sleep(10)  # Increased sleep time to reduce API load

        except Exception as e:
            logging.error(f"Error monitoring and logging orders: {str(e)}")
            await asyncio.sleep(20)

    async def log_trade(self, symbol, order_info):
        """
        Logs trade details including realized profit/loss and updates the trade records.
        """
        try:
            # Extract trade details from order_info
            order_id = order_info.get('orderId')
            status = order_info.get('status')
            filled_qty = order_info.get('executedQty')
            avg_fill_price = order_info.get('avgFillPrice')

            # Calculate realized profit/loss
            pnl = await self.calculate_unrealized_pnl(symbol)

            # Define the filename for trade logs
            directory = f"T:\\Trading BOT Crypto - Profit\\Profit"
            if not os.path.exists(directory):
                os.makedirs(directory)
            
            filename = os.path.join(directory, f"{symbol}_order.txt")

            # Write trade details to the file
            with open(filename, "a") as f:
                f.write(f"Order ID: {order_id}\n")
                f.write(f"Status: {status}\n")
                f.write(f"Quantity: {filled_qty}\n")
                f.write(f"Average Fill Price: {avg_fill_price}\n")
                f.write(f"Realized Profit/Loss: {pnl}\n")
                f.write(f"Timestamp: {datetime.datetime.now().isoformat()}\n\n")

            # Append trade record for Excel export
            self.trade_records.append({
                'Symbol': symbol,
                'Order ID': order_id,
                'Status': status,
                'Quantity': filled_qty,
                'Average Fill Price': avg_fill_price,
                'Realized Profit/Loss': pnl,
                'Timestamp': datetime.datetime.now().isoformat()
            })

        except Exception as e:
            logging.error(f"Error logging trade details for {symbol}: {str(e)}")
            await asyncio.sleep(20)

    async def calculate_unrealized_pnl(self, symbol):
        try:
            await self.initialize_session()
            
            if not self.session:
                logging.error("Session not initialized.")
                return

            async with self.session.get('https://api.binance.com/fapi/v2/positionRisk') as response:
                if response is None or response.status != 200:
                    logging.error(f"Failed to get a valid response from positionRisk endpoint.")
                    return
                
                positions = await response.json()
                positions = positions.get('positions', [])

            symbol_positions = [pos for pos in positions if pos['symbol'] == symbol and float(pos['positionAmt']) != 0]

            if not symbol_positions:
                logging.info(f"No ongoing futures positions for {symbol}.")
                return

            async with self.session.get(f'https://api.binance.com/fapi/v2/ticker/price', params={'symbol': symbol}) as response:
                if response is None or response.status != 200:
                    logging.error(f"Failed to get a valid response from ticker endpoint.")
                    return
                
                ticker = await response.json()
                current_price = float(ticker['price'])

            total_unrealized_pnl = 0.0

            for pos in symbol_positions:
                position_size = float(pos['positionAmt'])
                entry_price = float(pos['entryPrice'])
                unrealized_pnl = (current_price - entry_price) * position_size

                total_unrealized_pnl += unrealized_pnl

                logging.info(f"Position for {symbol}: Size {position_size}, Entry Price {entry_price}, Current Price {current_price}, Unrealized PNL {unrealized_pnl}")

            logging.info(f"Total Unrealized PNL for {symbol}: {total_unrealized_pnl}")

        except aiohttp.ClientError as e:
            logging.error(f"Network or client error while calculating unrealized PNL for {symbol}: {str(e)}", exc_info=True)
        except Exception as e:
            logging.error(f"Unexpected error while calculating unrealized PNL for {symbol}: {str(e)}", exc_info=True)

    async def periodically_update_unrealized_pnl(self, symbols, interval=60):
        while True:
            for symbol in symbols:
                await self.calculate_unrealized_pnl(symbol)
            await asyncio.sleep(interval)

    def update_tax_report(self):
        """
        Updates the tax report Excel file with the latest trade records.
        """
        try:
            # Convert trade records to DataFrame
            df = pd.DataFrame(self.trade_records)

            # Write to Excel file
            if os.path.exists(self.tax_report_file):
                with pd.ExcelWriter(self.tax_report_file, mode='a', if_sheet_exists='replace') as writer:
                    df.to_excel(writer, sheet_name='Trades', index=False)
            else:
                with pd.ExcelWriter(self.tax_report_file) as writer:
                    df.to_excel(writer, sheet_name='Trades', index=False)

            logging.info(f"Tax report updated successfully: {self.tax_report_file}")

        except Exception as e:
            logging.error(f"Error updating tax report: {str(e)}")
          
    async def log_open_order(self, symbol, order_id):
        """
        Logs an open order and saves its details to a text file.
        """
        try:
            self.open_orders[symbol] = {
                'order_id': order_id,
                'symbol': symbol,
                'status': 'open'
            }
            logging.info(f"Order logged: {self.open_orders[symbol]}")

            # Save order details to a text file
            filename = f"{symbol}_order.txt"
            with open(filename, "a") as f:
                f.write(f"Order ID: {order_id}\n")
                f.write(f"Symbol: {symbol}\n")
                f.write(f"Status: open\n\n")

        except Exception as e:
            logging.error(f"Error logging open order for {symbol}: {str(e)}")
            await asyncio.sleep(20)

    async def close(self):
        if self.session:
            await self.session.close()  # Properly close the session
            self.session = None  # Ensure it's set to None