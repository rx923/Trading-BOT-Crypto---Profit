from binance.client import AsyncClient
import os
import logging
from decimal import Decimal
import asyncio
import pandas as pd
from typing import Optional, Dict, List, Union, Optional
from dotenv import load_dotenv
import dotenv
from .BinanceManagerHandler import BinanceShortManager
# from data.Fetching_historical_OHCL_Values import BinanceDataProcessor
from binance.exceptions import BinanceAPIException, BinanceRequestException
# from Order_handlers.OrderPlacement import OrderPlacement
symbol_locks: Dict[List[str], asyncio.Lock] = {}

logging.basicConfig(level=logging.INFO)
dotenv.load_dotenv()  # Load environment variables from a .env file

# Initialize the Binance client with appropriate API keys
api_key = os.getenv('BINANCE_API_KEY')
api_secret = os.getenv('BINANCE_API_SECRET')
# Use await 
# client = AsyncClient.create(api_key, api_secret)  

# Set the event loop policy to SelectorEventLoop for compatibility with aiodns
if os.name == 'nt':  # Check if the OS is Windows
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

class OrderManagerTakeProfit:
    def __init__(self, client, symbol, api_key=api_key, api_secret=api_secret):
        self.client = client
        self.symbols = symbol
        self.api_key = api_key
        self.running_positions = None
        self.api_secret = api_secret

        # self.Fetching_Running_Positions = OrderPlacement(client, symbols)
        # self._fetching_klines_ = BinanceDataProcessor(self, client, order_placement=None)
        self.binance_manager = BinanceShortManager(self, symbol, client, api_key)  # Initialize with client
        self.running_positions = None  # Initialize it first

        logging.info("OrderManagerTakeProfit class initialized successfully")

    async def initialize_running_positions(self):
        # self.running_positions = await self.binance_manager.fetch_running_positions()  # Fetch and assign positions asynchronously
        # Call check_ongoing_positions after fetching running positions
        ongoing_positions = await self.binance_manager.check_ongoing_positions()
        return ongoing_positions  # Return or process ongoing positions as needed


    async def initialize_client(self, api_key=api_key, api_secret=api_secret):
        """ Initialize Binance async client """
        try:
            self.client = await AsyncClient.create(api_key=api_key, api_secret=api_secret)
            logging.info("Binance async client initialized")
        except Exception as e:
            logging.error(f"Error initializing Binance async client: {e}")

    @staticmethod
    async def fetch_klines(client, symbol, interval='1m', start_time='2024-08-01', end_time=None):
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
            klines = client.futures_klines(
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



    async def calculate_dynamic_take_profit_short(self, symbol):
    try:
        # Ensure self.running_positions is properly initialized
        if self.running_positions is None:
            self.running_positions = {}

        # Check if the symbol exists in running_positions
        if symbol not in self.running_positions:
            logging.error(f"No ongoing positions found for {symbol}. Cannot calculate dynamic take profit.")
            return None

        position = self.running_positions[symbol]

        # Retrieve entry_price and position_quantity
        entry_price = position.get('entryPrice')
        position_quantity = position.get('positionAmt')

        # Separate validation for entry_price
        if entry_price is None:
            logging.error(f"Failed to retrieve 'entryPrice' for {symbol}.")
            return None
        else:
            entry_price = Decimal(entry_price)  # Ensure it's a Decimal

        # Separate validation for position_quantity
        if position_quantity is None:
            logging.error(f"Failed to retrieve 'positionAmt' for {symbol}.")
            return None
        else:
            position_quantity = Decimal(position_quantity)  # Ensure it's a Decimal

        # Check if entry_price and position_quantity are positive
        if entry_price <= 0:
            logging.error(f"Invalid entry price ({entry_price}) for {symbol}. Cannot calculate dynamic take profit.")
            return None

        if position_quantity <= 0:
            logging.error(f"Invalid position quantity ({position_quantity}) for {symbol}. Cannot calculate dynamic take profit.")
            return None

        # Calculate take-profit levels and quantities
        take_profit_levels = []
        take_profit_quantities = []
        remaining_quantity = position_quantity

        # Process each take-profit level and log the calculation
        for i in range(1, 11):  # Create 10 take-profit levels
            take_profit_ratio = Decimal('0.002') * i  # Example: 0.2% decrease for each level
            take_profit_level = entry_price * (Decimal('1') - take_profit_ratio)  # Lower than entry price
            take_profit_levels.append(take_profit_level)

            # Calculate quantity to close at each level
            if remaining_quantity > 0:
                quantity_to_close = position_quantity * Decimal('0.1')  # 10% of the total position
                remaining_quantity -= quantity_to_close
                take_profit_quantities.append(quantity_to_close)
            else:
                take_profit_quantities.append(Decimal('0'))

            # Log the current level and calculations for the symbol
            logging.info(f"[{symbol}] Level {i}: Take profit at {take_profit_level:.6f}, Quantity to close: {quantity_to_close:.6f}")

        # Log the final results
        logging.info(f"[{symbol}] Final take profit levels: {take_profit_levels}")
        logging.info(f"[{symbol}] Final take profit quantities: {take_profit_quantities}")

        return {
            'take_profit_levels': take_profit_levels,
            'take_profit_quantities': take_profit_quantities
        }

    except Exception as e:
        logging.error(f"Error calculating dynamic take profit ratios for {symbol}: {str(e)}")
        return None

    finally:
        # Ensure that any necessary cleanup or final logging occurs
        logging.info(f"Completed calculation attempt for {symbol}.")


    async def place_take_profit_order_short(self, symbol: str, entry_price: Decimal, take_profit_levels: list[Decimal], initial_quantity: Decimal) -> bool:
        try:
            if not take_profit_levels or len(take_profit_levels) < 1:
                logging.warning(f"Insufficient take profit levels provided for {symbol}. Skipping order placement.")
                return False

            number_of_levels = len(take_profit_levels)
            quantity_per_level = initial_quantity / Decimal(number_of_levels)

            logging.info(f"Placing {number_of_levels} take profit orders for {symbol}. Quantity per order: {quantity_per_level}")

            order_tasks = []
            remaining_quantity = initial_quantity

            for i, take_profit_price in enumerate(take_profit_levels):
                if i == (number_of_levels - 1):
                    order_quantity = remaining_quantity
                else:
                    order_quantity = quantity_per_level

                # Ensure correct price-based exit configuration
                if entry_price and take_profit_price < Decimal(entry_price):
                    order_quantity = min(order_quantity, remaining_quantity) * Decimal('0.05')

                order_task = self.client.futures_create_order(
                    symbol=symbol,
                    type='LIMIT',  # Use LIMIT for precise price setting
                    side='BUY',  # To close a short position
                    positionSide='SHORT',
                    quantity=str(order_quantity),
                    price=str(take_profit_price),  # Limit price for take profit / You may set stopPrice if it's a stop limit order
                    timeInForce='GTC',  # Good till canceled
                    reduceOnly=True,  # Ensures order only reduces the position
                    postOnly=False,  # Set this based on your preference
                    
                    closePosition=False  # Change this if you want to close the position entirely
                )
                order_tasks.append(order_task)

                logging.info(f"Placing take profit order {i+1} for {symbol} at price {take_profit_price} with quantity {order_quantity}")

                remaining_quantity -= order_quantity

                if remaining_quantity <= 0:
                    break

            # Execute all order tasks concurrently
            order_responses = await asyncio.gather(*order_tasks)

            for i, order_response in enumerate(order_responses):
                if 'orderId' in order_response:
                    logging.info(f"Take profit order {i+1} placed successfully for {symbol}: {order_response}")
                else:
                    logging.error(f"Failed to place take profit order {i+1} for {symbol}: {order_response}")
                    return False

            return True
        except Exception as e:
            logging.error(f"Error placing take profit orders for {symbol}: {e}", exc_info=True)
            return False
        
        finally:
            # Ensure final logging or cleanup
            logging.info(f"Completed take profit order placement attempt for {symbol}.")


    async def place_take_profit_order_wrapper(self, symbol, client, entry_price: Decimal, initial_quantity: Decimal):
        try:
            if entry_price <= 0 or initial_quantity <= Decimal('0'):
                logging.error(f"Invalid entry_price ({entry_price}) or initial_quantity ({initial_quantity}) for {symbol}.")
                return False

            logging.info(f"Calculating dynamic take profit for {symbol} with entry price {entry_price} and quantity {initial_quantity}")

            # Pass client and symbol properly
            take_profit_info = await self.calculate_dynamic_take_profit_short(symbol)

            if take_profit_info is None or 'take_profit_levels' not in take_profit_info:
                logging.error(f"Failed to calculate dynamic take profit levels for {symbol}. Aborting take profit order placement.")
                return False

            take_profit_levels = [Decimal(level) for level in take_profit_info['take_profit_levels']]
            logging.info(f"Take profit levels for {symbol}: {take_profit_levels}")

            result = await self.place_take_profit_order_short(
                symbol=symbol,
                entry_price=entry_price,
                take_profit_levels=take_profit_levels,
                initial_quantity=initial_quantity
            )

            if result:
                logging.info(f"Take profit order placed successfully for {symbol}")
            else:
                logging.error(f"Failed to place take profit order for {symbol}")

            return result

        except Exception as e:
            logging.error(f"Error placing take profit orders for {symbol}: {e}", exc_info=True)
            return False
        finally:
            # Ensure final logging or cleanup
            logging.info(f"Completed take profit order wrapper process for {symbol}.")



    # async def calculate_dynamic_take_profit_long(self, symbol: str, entry_price: Decimal, is_long: bool = True, df_1m: Optional[pd.DataFrame] = None):
    #     try:
    #         # Ensure async lock for the symbol
    #         async with symbol_locks.get(symbol, asyncio.Lock()):
                
    #             # Fetch klines if no dataframe is provided
    #             if df_1m is None or df_1m.empty:
    #                 klines = await self.fetch_klines(client, symbol, interval='1m', start_time='2024-08-01', end_time=None)
    #                 if klines is None:
    #                     logging.warning(f"Failed to fetch klines for {symbol}.")
    #                     return None
    #                 df_1m = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

    #             recent_high_1m = df_1m['high'].max()
    #             recent_low_1m = df_1m['low'].min()

    #             if is_long:
    #                 current_price = entry_price
    #             else:
    #                 logging.error(f"Invalid position type for {symbol}. Cannot calculate dynamic take profit for long.")
    #                 return None

    #             if current_price is None:
    #                 logging.error(f"Failed to fetch current price for {symbol}. Cannot calculate dynamic take profit.")
    #                 return None

    #             take_profit_levels = []
    #             for i in range(1, 11):
    #                 take_profit_ratio = 0.005 * i  # Example: 0.5% increase per level for long positions
    #                 take_profit_level = current_price * Decimal(1 + take_profit_ratio)  # Levels are above entry price
    #                 take_profit_levels.append(take_profit_level)

    #             logging.debug(f"Recent high 1m: {recent_high_1m}, Recent low 1m: {recent_low_1m}")
                
    #             return {
    #                 'take_profit_levels': take_profit_levels,
    #                 'recent_high_1m': recent_high_1m,
    #                 'recent_low_1m': recent_low_1m
    #             }

    #     except Exception as e:
    #         logging.error(f"Error calculating dynamic take profit levels for {symbol}: {str(e)}")
    #         return None
        

    # async def place_take_profit_order_long(self, symbol: str, entry_price: float, take_profit_levels: List[Decimal], initial_quantity: Decimal):
    #     try:
    #         if not take_profit_levels or len(take_profit_levels) < 4:
    #             logging.warning(f"Insufficient take profit levels provided for {symbol}. Skipping order placement.")
    #             return False

    #         order_tasks = []
    #         remaining_quantity = initial_quantity

    #         for i, take_profit_price in enumerate(take_profit_levels, start=1):
    #             order_quantity = remaining_quantity * Decimal('0.1')  # 10% of remaining quantity

    #             # Ensure take profit is above the entry price for long positions
    #             if entry_price and take_profit_price > Decimal(entry_price):
    #                 order_quantity = remaining_quantity * Decimal('0.1')  # 10% per level as the price increases

    #             order_task = self.client.futures_create_order(
    #                 symbol=symbol,
    #                 type='TAKE_PROFIT_MARKET',
    #                 side='SELL',  # Sell to close a long position at a higher price
    #                 quantity=str(order_quantity),
    #                 stopPrice=str(take_profit_price),
    #                 closePosition=True
    #             )
    #             order_tasks.append(order_task)

    #             logging.info(f"Placing take profit order {i} for {symbol} at price {take_profit_price}")

    #             remaining_quantity -= order_quantity

    #         # Gather all order responses
    #         order_responses = await asyncio.gather(*order_tasks)

    #         for i, order_response in enumerate(order_responses, start=1):
    #             if 'orderId' in order_response:
    #                 logging.info(f"Take profit order {i} placed successfully for {symbol}: {order_response}")
    #             else:
    #                 logging.error(f"Failed to place take profit order {i} for {symbol}: {order_response}")
    #                 return False

    #         return True

    #     except Exception as e:
    #         logging.error(f"Error placing take profit orders for {symbol}: {e}")
    #         return False

    async def close(self):
        """Close the Binance client session."""
        if self.client:
            await self.client.close_connection()
            logging.info("Binance client connection closed.")
