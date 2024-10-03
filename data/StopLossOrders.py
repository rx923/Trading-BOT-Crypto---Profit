import logging
import os
from binance.exceptions import BinanceAPIException
from typing import Tuple, Optional, Callable, Any, List
from binance import AsyncClient
from decimal import Decimal, ROUND_DOWN
from colorama import Fore, Style, Back
from dotenv import load_dotenv
import aiohttp
import asyncio
import dotenv
from decimal import ROUND_UP
logging.basicConfig(level=logging.INFO)
from logger_config import get_logger


# Get the logger for this module
logger = get_logger(__name__, blue_background=True)

dotenv.load_dotenv()  # Load environment variables from a .env file

api_key = os.getenv('BINANCE_API_KEY')
api_secret = os.getenv('BINANCE_API_SECRET')
# client = AsyncClient.create(api_key=api_key, api_secret=api_secret)





class StopLossOrders:
    def __init__(self, client, symbol):
        self.api_key = os.getenv('BINANCE_API_KEY')
        self.api_secret = os.getenv('BINANCE_API_SECRET')
        self.client = client
        self.symbol = symbol
        logger.info("StopLossOrders class initialized successfully")


    async def initialize_client(self):
        try:
            self.client = await AsyncClient.create(api_key=api_key, api_secret=api_secret)
            logging.info("Binance async client initialized")
        except Exception as e:
            logging.error(f"Error initializing Binance async client: {e}")
            await asyncio.sleep(20)

    async def fetch_symbol_rules(self):
        #To add closure of the aiohttp connection and retry mechanism and clearing of the logs later on
        url = "https://fapi.binance.com/fapi/v1/exchangeInfo"
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                data = await response.json()
                symbols_info = data['symbols']
                symbol_rules = {}
                
                for symbol_data in symbols_info:
                    symbol = symbol_data['symbol']
                    min_trade_amount = Decimal(symbol_data['filters'][1]['minQty'])
                    min_price_movement = Decimal(symbol_data['filters'][0]['tickSize'])
                    price_precision = symbol_data['pricePrecision']
                    min_notional_value = Decimal(symbol_data['filters'][3]['minNotional'])
                    max_market_order_amount = Decimal(symbol_data['filters'][2]['maxQty'])
                    max_limit_order_amount = Decimal(symbol_data['filters'][2]['maxQty'])

                    symbol_rules[symbol] = {
                        'min_trade_amount': min_trade_amount,
                        'min_price_movement': min_price_movement,
                        'price_precision': price_precision,
                        'min_notional_value': min_notional_value,
                        'max_market_order_amount': max_market_order_amount,
                        'max_limit_order_amount': max_limit_order_amount
                    }

                return symbol_rules

    async def get_symbol_precision(self, symbol: List[str]) -> Tuple[int, int]:
        try:
            logging.info(f"Fetching precision for symbol: {symbol}")
            exchange_info = await self.client.get_exchange_info()
            symbol_info = next((s for s in exchange_info['symbols'] if s['symbol'] == symbol), None)

            if symbol_info:
                price_precision = int(symbol_info.get('quotePrecision', 0))
                quantity_precision = int(symbol_info.get('baseAssetPrecision', 0))
                logging.info(f"Fetched precision for {symbol}: Price Precision = {price_precision}, Quantity Precision = {quantity_precision}")
                return price_precision, quantity_precision
            else:
                logging.error(f"Symbol {symbol} not found in exchange info.")
                return 0, 0

        except Exception as e:
            logging.error(f"Error fetching symbol precision for {symbol}: {e}")
            await asyncio.sleep(20)
            return 0, 0
            
    async def process_order(self, symbol: List[str], entry_price: Decimal, total_position_size: Decimal):
        async def fetch_precision():
            return await self.get_symbol_precision(symbol)

        try:
            price_precision, quantity_precision = await self.retry_operation(fetch_precision)

            if price_precision == 0 or quantity_precision == 0:
                logging.error(f"[{symbol}] Invalid price or quantity precision (price: {price_precision}, quantity: {quantity_precision}). Skipping order.")
                await asyncio.sleep(20)
                return None

            logging.info(f"[{symbol}] Price precision: {price_precision}, Quantity precision: {quantity_precision}")

            entry_price = entry_price.quantize(Decimal(f'1.{"0" * price_precision}'), rounding=ROUND_UP)
            quantity = total_position_size.quantize(Decimal(f'1.{"0" * quantity_precision}'), rounding=ROUND_UP)

            if quantity <= Decimal('0'):
                logging.error(f"[{symbol}] Invalid quantity calculated: {quantity}. Skipping order.")
                await asyncio.sleep(20)
                return None
            
            quantity_str = str(quantity)
            return entry_price, quantity_str

        except Exception as e:
            logging.error(f"[{symbol}] Error processing order: {e}")
            await asyncio.sleep(20)
            return None

    async def fetch_order_status(self, symbol: List[str], order_id: Optional[int] = None) -> dict | list | None:
        if self.client is None:
            logging.error("Binance client is not initialized.")
            return None

        try:
            if order_id is not None:
                # Fetch specific order status by order_id
                return await self.client.futures_get_order(symbol=symbol, orderId=order_id)
            else:
                # Fetch all ongoing orders for the symbol
                open_orders = await self.client.futures_get_open_orders(symbol=symbol)
                if not open_orders:
                    logging.info(f"No ongoing orders for {symbol}.")
                    return []

                # Process each order to fetch detailed status
                detailed_orders = []
                for order in open_orders:
                    order_id = order['orderId']
                    order_status = await self.client.futures_get_order(symbol=symbol, orderId=order_id)
                    detailed_orders.append(order_status)
                    logging.info(f"Order status for {symbol} (Order ID: {order_id}): {order_status}")

                return detailed_orders

        except BinanceAPIException as e:
            logging.error(f"API error fetching order status for {symbol} (Order ID: {order_id}): {e.message}")
        except Exception as e:
            logging.error(f"Error fetching order status for {symbol} (Order ID: {order_id}): {str(e)}")
        return None

    async def retry_operation(self, operation: Callable[[], Any], retries: int = 3, delay: int = 20) -> Any:
        for attempt in range(retries):
            try:
                return await operation()
            except Exception as e:
                logging.error(f"Attempt {attempt + 1} failed: {e}")
                if attempt < retries - 1:
                    await asyncio.sleep(delay)
                else:
                    logging.error("Max retries reached. Operation failed.")
                    raise e

    async def fetch_position_details(self, symbol: List[str]) -> dict:
        async def fetch():
            positions = await self.client.futures_position_information(symbol=symbol)
            for position in positions:
                if position['symbol'] == symbol and position['positionSide'] == 'SHORT':
                    return {
                        'entryPrice': Decimal(position['entryPrice']),
                        'quantity': Decimal(position['positionAmt'])
                    }
            return {}

        try:
            return await self.retry_operation(fetch)
        except Exception as e:
            logging.error(f"Error fetching position details for {symbol}: {e}", exc_info=True)
            return {}
        
    async def fetch_latest_candle(self, symbol, interval: str = '1m') -> Decimal | None:
        try:
            klines = await self.client.get_klines(symbol=symbol, interval=interval, limit=1)
            if klines:
                latest_close = klines[0][4]  
                return Decimal(str(latest_close))
            else:
                logging.error(f"No kline data available for {symbol}")
                await asyncio.sleep(20)
                return None
        except Exception as e:
            logging.error(f"Error fetching latest candle for {symbol}: {e}")
            await asyncio.sleep(20)
            return None

    async def calculate_dynamic_stop_loss_short(
        self,
        symbol: List[str],
        order_id: Optional[int] = None,
        risk_level: Decimal = Decimal('0.02'),
        max_price_diff: Decimal = Decimal('0.1')
    ) -> Optional[Decimal]:
        try:
            async def fetch_data():
                # Fetch order status
                order_data = await self.fetch_order_status(symbol, order_id)
                if order_data is None:
                    logging.error(f"Order data for {symbol} is None.")
                    return None

                # Extract entry price
                entry_price = None
                if order_id is not None:
                    if isinstance(order_data, dict) and 'position' in order_data:
                        entry_price = Decimal(order_data['position']['entryPrice'])
                    else:
                        logging.error(f"Failed to extract entry price for order ID {order_id} in {symbol}.")
                        return None
                else:
                    if isinstance(order_data, list) and len(order_data) > 0:
                        position = order_data[0].get('position')
                        if position:
                            entry_price = Decimal(position['entryPrice'])
                        else:
                            logging.error(f"No position found in order data list for {symbol}.")
                            return None
                    else:
                        logging.error(f"Order data list is empty or invalid for {symbol}.")
                        return None

                # Check if entry price was successfully set
                if entry_price is None:
                    logging.error(f"Entry price is None for {symbol}.")
                    return None
                logging.info(f"Entry price for {symbol} is {entry_price}.")

                # Fetch current price
                current_price = await self.fetch_latest_candle(symbol)
                if current_price is None:
                    logging.error(f"Failed to fetch current price for {symbol}.")
                    return None

                # Get symbol precision
                price_precision, _ = await self.get_symbol_precision(symbol)
                if price_precision == 0:
                    logging.error(f"Price precision for {symbol} is 0, invalid precision.")
                    return None

                # Convert current price to Decimal and log
                current_price = Decimal(str(current_price))
                logging.info(f"Current price for {symbol} is {current_price}.")

                # Calculate the stop loss price
                stop_loss_price = entry_price - (entry_price * risk_level)
                logging.info(f"Initial stop loss price for {symbol} calculated as {stop_loss_price}.")

                # Check max price difference condition
                if abs(current_price - stop_loss_price) > max_price_diff:
                    stop_loss_price = entry_price - max_price_diff
                    logging.info(f"Stop loss price for {symbol} adjusted to {stop_loss_price} based on max price difference.")

                # Adjust stop loss price to required precision
                stop_loss_price = stop_loss_price.quantize(Decimal('1.' + '0' * price_precision), rounding=ROUND_DOWN)
                logging.info(f"Final stop loss price for {symbol} is {stop_loss_price}.")

                return stop_loss_price

            # Retry operation to handle transient errors
            return await self.retry_operation(fetch_data)

        except Exception as e:
            logging.error(f"Error calculating stop loss for {symbol}: {e}")
            await asyncio.sleep(20)
            return None

    async def place_stop_loss_order_short(
        self,
        symbol: List[str],
        stop_loss_price: Decimal,
        quantity: Decimal
    ) -> Optional[dict]:
        if self.client is None:
            logging.error("Binance client is not initialized.")
            await asyncio.sleep(20)
            return None

        try:
            logging.info(f"Attempting to place stop loss order for {symbol}.")
            logging.debug(f"Stop loss price: {stop_loss_price}, Quantity: {quantity}")

            stop_loss_price_str = str(stop_loss_price)
            quantity_str = str(quantity)

            stop_loss_order = await self.client.futures_create_order(
                symbol=symbol,
                side='BUY',
                quantity=quantity_str,
                type='STOP_MARKET',
                stopPrice=stop_loss_price_str,
                reduceOnly=True,
                closePosition=True,
                positionSide='SHORT',
                timeInForce='IMMEDIATE_OR_CANCEL'
            )

            logging.info(f"Stop loss order response: {stop_loss_order}")

            if 'orderId' in stop_loss_order:
                order_id = stop_loss_order['orderId']
                logging.info(f"Stop loss order placed with ID: {order_id}")

                order_status = await self.fetch_order_status(symbol, order_id)
                logging.info(f"Stop loss order status: {order_status}")

                return stop_loss_order
            else:
                logging.error("Stop loss order placement failed. No order ID returned.")
                return None

        except Exception as e:
            logging.error(f"Exception placing stop loss order for {symbol}: {str(e)}", exc_info=True)
            await asyncio.sleep(20)
            return None

    async def check_stop_loss_condition(self, symbol, unrealized_pnl, position_details):
        try:
            if unrealized_pnl < -0.1 * position_details['initial_unrealized_pnl']:
                return True
            return False
        except Exception as e:
            logging.error(f"Error checking stop loss condition for {symbol}: {e}")
            await asyncio.sleep(20)
            return False

    async def close(self):
        """Close the Binance client session."""
        if self.client:
            await self.client.close_connection()
            logging.info("Binance client connection closed.")