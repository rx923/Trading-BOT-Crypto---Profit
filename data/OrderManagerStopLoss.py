from binance.client import AsyncClient
import os
import logging
import uuid
import asyncio
import dotenv
dotenv.load_dotenv()  # Load environment variables from a .env file


# Correct relative imports within the data directory
from .OrderManagerTakeProfit import OrderManagerTakeProfit

# from .PositionSizer import PositionSizer
from .StopLossOrders import StopLossOrders


from colorama import Fore, Style, Back
# import datetime


# Set the event loop policy to SelectorEventLoop for compatibility with aiodns
if os.name == 'nt':  # Check if the OS is Windows
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Load API credentials from environment variables
api_key = os.getenv('BINANCE_API_KEY')
api_secret = os.getenv('BINANCE_API_SECRET')

# client = AsyncClient.create(api_key, api_secret)



class ColoredFormatter(logging.Formatter):
    def format(self, record):
        # Set colors based on log level
        if record.levelno == logging.INFO:
            color = Fore.GREEN
        elif record.levelno == logging.WARNING:
            color = Fore.YELLOW
        elif record.levelno == logging.ERROR:
            color = Fore.RED
        else:
            color = Style.RESET_ALL

        # Format the message with color
        formatted_message = f"{color}{record.levelname} - {record.getMessage()}{Style.RESET_ALL}"

        return formatted_message

# Configure the root logger
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
formatter = ColoredFormatter()
handler.setFormatter(formatter)
logger.addHandler(handler)

# Configure the root logger
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
formatter = ColoredFormatter()
handler.setFormatter(formatter)
logger.addHandler(handler)



class OrderManagerStopLoss:
    def __init__(self, client: AsyncClient):
        self.api_key = os.getenv('BINANCE_API_KEY')
        self.api_secret = os.getenv('BINANCE_API_SECRET')
        self.client = client
        self.active_orders = set()
        self.closed_orders = set()

    async def initialize_client(self, api_key, api_secret):
        """ Initialize Binance async client """
        try:
            self.client = await AsyncClient.create(api_key=api_key, api_secret=api_secret)
            logging.info("Binance async client initialized")
        except Exception as e:
            logging.error(f"Error initializing Binance async client: {e}")
    
    # async def place_take_profit_order(self, symbol: str, order_id: str, position_info: dict, unrealized_pnl: float, current_candle_close: float):
    #     try:
    #         # Example: Calculate order quantities and prices based on strategy
    #         initial_quantity = position_info['initial_quantity']
    #         entry_price = position_info['entry_price']

    #         # Calculate dynamic take profit prices
    #         # take_profit_prices = await OrderManagerTakeProfit.calculate_dynamic_take_profit(self, symbol, entry_price, initial_quantity)

    #         # if take_profit_prices is None:
    #         #     logging.error(f"Failed to calculate dynamic take profit prices for {symbol}. Aborting take profit order placement.")
    #         #     return

    #         # Place take profit orders based on the prices calculated
    #         for i, take_profit_price in enumerate(take_profit_prices):
    #             order_quantity = initial_quantity * 0.1  # Adjust as per your strategy
    #             order = await self.client.futures_create_order(
    #                 symbol=symbol,
    #                 type='TAKE_PROFIT_MARKET',
    #                 side='BUY',  # Assuming closing a short position
    #                 quantity=order_quantity,
    #                 stopPrice=take_profit_price,
    #                 closePosition=True,
    #                 newClientOrderId=str(uuid.uuid4()),
    #                 stopClientOrderId=order_id  # Linking take profit order to initial order
    #             )

    #             if 'orderId' in order:
    #                 logging.info(f"Take profit order {i+1} placed for {symbol} at price {take_profit_price}: {order}")
    #                 self.active_orders.add(order['orderId'])
    #             else:
    #                 logging.error(f"Failed to place take profit order {i+1} for {symbol}: {order}")

    #         # Mark take profit orders as placed
    #         position_info['take_profit_orders_placed'] = True

    #     except Exception as e:
    #         logging.error(f"Error placing take profit orders for {symbol}: {e}")

    # async def cancel_order(self, symbol: str, order_id: str):
    #     try:
    #         # Cancel an active order
    #         await self.client.futures_cancel_order(
    #             symbol=symbol,
    #             orderId=order_id
    #         )
    #         self.active_orders.remove(order_id)  # Remove order ID from active orders set after cancellation
    #         logging.info(f"Cancelled order {order_id} for {symbol}")

    #         # Add order ID to closed orders set
    #         self.closed_orders.add(order_id)

    #         # Save closed orders to a file
    #         await self.save_orders_to_file()

    #     except Exception as e:
    #         logging.error(f"Error cancelling order {order_id} for {symbol}: {e}")

    # async def close_position(client, symbol, position_size):
    #     try:
    #         from .ProfitLossMonitor import ProfitLossMonitor
    #         # Fetch current unrealized PNL data
    #         pnl_data = await ProfitLossMonitor.monitor_unrealized_pnl(client, symbol)
    #         if not pnl_data:
    #             logging.warning(f"No position data found for {symbol}. Cannot close position.")
    #             return

    #         unrealized_pnl_percentage = pnl_data['unrealized_pnl_percentage']

    #         # Get current position size if not provided
    #         if position_size is None:
    #             position_size = PositionSizer.calculate_position_size(symbol)
    #             if position_size <= 0:
    #                 logging.warning(f"No position size found for {symbol}.")
    #                 return

    #         # Condition to close 25% of position if unrealized PNL reaches 25%
    #         if unrealized_pnl_percentage >= 0.25:
    #             await OrderManagerStopLoss.close_position(client, symbol, position_size=5, fraction=0.25)
    #             logging.info(f"Partially closed 25% position for {symbol} due to high unrealized PNL: {unrealized_pnl_percentage:.2%}")

    #         # Condition to close remaining 75% of position if unrealized PNL reaches 50%
    #         elif unrealized_pnl_percentage >= 0.50:
    #             await OrderManagerStopLoss.close_position(client, symbol, position_size=5, fraction=0.75)
    #             logging.info(f"Closed remaining 75% position for {symbol} due to high unrealized PNL: {unrealized_pnl_percentage:.2%}")

    #         # Condition to trigger stop-loss if unrealized PNL reaches -5%
    #         elif unrealized_pnl_percentage < -0.05:
    #             await StopLossOrders.trigger_stop_loss(client, symbol)
    #             logging.info(f"Closed position for {symbol} due to maximum allowable loss reached")

    #         else:
    #             logging.info(f"No action taken for {symbol}. Unrealized PNL: {unrealized_pnl_percentage:.2%}")

    #     except Exception as e:
    #         logging.error(f"Error closing position for {symbol}: {e}")

    async def close(self):
        await self.client.close_connection()