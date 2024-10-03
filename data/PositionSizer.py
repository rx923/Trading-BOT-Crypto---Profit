# import logging
# import os
# from binance import AsyncClient
# from decimal import Decimal

# class PositionSizer:
#     def __init__(self):
#         self.api_key = os.getenv('BINANCE_API_KEY')
#         self.api_secret = os.getenv('BINANCE_API_SECRET')
#         self.logger = logging.getLogger(__name__)
        
#         # Define symbols with their rules
#         self.symbols_with_special_rules = {
#             'BTCUSDT': {'price_precision': 6, 'quantity_precision': 2},
#             'ETHUSDT': {'price_precision': 4, 'quantity_precision': 3},
#             'RVNUSDT': {'price_precision': 2, 'quantity_precision': 2}
#         }

#     async def initialize_client(self):
#         """ Initialize Binance async client """
#         try:
#             self.client = await AsyncClient.create(api_key=self.api_key, api_secret=self.api_secret)
#             self.logger.info("Binance async client initialized")
#         except Exception as e:
#             self.logger.error(f"Error initializing Binance async client: {e}")

#     async def calculate_position_size(self, symbol, current_price, amount_usdt, price_precision=None, quantity_precision=None,
#                                       balance=None, initial_balance=5, max_leverage=20, trade_history=None):
#         try:
#             # Validate input
#             if current_price is None or current_price <= 0:
#                 raise ValueError(f"Invalid price for {symbol}: {current_price}. Cannot calculate position size.")
            
#             if amount_usdt is None or amount_usdt <= 0:
#                 raise ValueError("Amount in USDT (amount_usdt) must be provided and greater than zero.")

#             # Determine which balance to use: current_balance or initial_balance
#             balance_to_use = balance if balance is not None else initial_balance

#             # Fetch symbol-specific rules if available
#             if symbol in self.symbols_with_special_rules:
#                 price_precision = self.symbols_with_special_rules[symbol]['price_precision']
#                 quantity_precision = self.symbols_with_special_rules[symbol]['quantity_precision']

#             # Calculate the base position size considering price precision
#             base_position_size = (amount_usdt * max_leverage) / current_price

#             # Round the base position size to the desired quantity precision
#             position_size = round(base_position_size, quantity_precision)

#             # Ensure the position size does not exceed the maximum position size
#             max_position_size_units = round((amount_usdt * 0.1) / current_price, quantity_precision)
#             position_size = min(position_size, max_position_size_units)

#             # Ensure the position size does not exceed 10% of the balance
#             max_units_from_balance = round((balance_to_use * 0.1) / current_price, quantity_precision)
#             position_size = min(position_size, max_units_from_balance)

#             self.logger.info(f"Calculated position size for {symbol}: {position_size} units at price {current_price}")
#             self.logger.debug(f"Price precision: {price_precision}, Quantity precision: {quantity_precision}")

#             if trade_history:
#                 self.logger.debug(f"Trade history length: {len(trade_history)}")

#             return position_size

#         except Exception as e:
#             self.logger.error(f"Error calculating position size for {symbol}: {str(e)}")
#             return None

#     async def fetch_open_positions(self):
#         """
#         Fetches all open positions for the account.

#         Returns:
#             List of open positions with relevant details.
#         """
#         try:
#             positions = await self.client.futures_position_information()
#             open_positions = []

#             for position in positions:
#                 if float(position['positionAmt']) != 0:  # Filter out positions with zero quantity
#                     open_positions.append({
#                         'symbol': position['symbol'],
#                         'entry_price': float(position['entryPrice']),
#                         'position_amt': float(position['positionAmt']),
#                         'unrealized_pnl': float(position['unRealizedProfit']),
#                         'liquidation_price': float(position['liquidationPrice']),
#                         'leverage': int(position['leverage']),
#                         'position_side': position['positionSide']
#                     })

#             self.logger.info(f"Fetched {len(open_positions)} open positions.")
#             return open_positions

#         except Exception as e:
#             self.logger.error(f"Error fetching open positions: {e}")
#             return None

#     async def calculate_total_unrealized_pnl(self):
#         """
#         Calculate the total unrealized profit and loss for all open positions.

#         Returns:
#             Total unrealized PNL in USDT.
#         """
#         try:
#             open_positions = await self.fetch_open_positions()
#             total_unrealized_pnl = 0.0

#             if open_positions:
#                 for position in open_positions:
#                     pnl = position['unrealized_pnl']
#                     total_unrealized_pnl += pnl

#                     self.logger.info(f"Position {position['symbol']}: Unrealized PNL: {pnl:.4f} USDT")

#             self.logger.info(f"Total Unrealized PNL: {total_unrealized_pnl:.4f} USDT")
#             return total_unrealized_pnl

#         except Exception as e:
#             self.logger.error(f"Error calculating total unrealized PNL: {e}")
#             return None

#     async def calculate_required_margin(self, symbol, quantity, leverage):
#         """
#         Calculate the required margin for opening a position.

#         Parameters:
#             symbol: Trading symbol (e.g., 'BTCUSDT').
#             quantity: Quantity of the asset to trade.
#             leverage: Leverage to use for the trade.

#         Returns:
#             Required margin in quote asset.
#         """
#         try:
#             # price_info = await self.client.futures_symbol_price_ticker(symbol=symbol)
#             # price = float(price_info['price'])

#             # Accessing quantity_precision and price_precision
#             quantity_precision = self.symbols_with_special_rules[symbol]['quantity_precision']
#             price_precision = self.symbols_with_special_rules[symbol]['price_precision']

#             # Calculate notional value of the trade
#             # notional_value = quantity * price

#             # Calculate margin required
#             margin_ratio = 1 / leverage
#             margin_multiplier = 1.1  # Adjust as per your risk management
#             # margin = notional_value * margin_ratio * margin_multiplier

#             # self.logger.info(f"Required margin for {symbol} with quantity {quantity} and leverage {leverage}: {margin:.4f} USDT")

#             # return margin

#         except Exception as e:
#             self.logger.error(f"Error calculating required margin for {symbol}: {e}")
#             return None

#     async def fetch_position_details(self, symbol):
#         """
#         Fetch details of an open position for the specified symbol.

#         Returns:
#             Dictionary containing position details.
#         """
#         try:
#             positions = await self.client.futures_position_information(symbol=symbol)
#             for position in positions:
#                 if position['symbol'] == symbol and float(position['positionAmt']) != 0:
#                     return {
#                         'symbol': symbol,
#                         'position_status': position['positionSide'],
#                         'entry_price': float(position['entryPrice']),
#                         'current_qty': float(position['positionAmt']),
#                         'unrealized_pnl': float(position['unRealizedProfit']),
#                         'liquidation_price': float(position['liquidationPrice'])
#                     }

#             self.logger.info(f"No open position found for {symbol}")
#             return None

#         except Exception as e:
#             self.logger.error(f"Error fetching position details for {symbol}: {e}")
#             return None

#     async def place_stop_loss_order(self, symbol, stop_loss_price, quantity):
#         """
#         Places a stop loss order for the specified symbol.

#         Parameters:
#             symbol: Trading symbol (e.g., 'BTCUSDT').
#             stop_loss_price: Stop loss price.
#             quantity: Quantity of the asset to trade.

#         Returns:
#             Stop loss order response.
#         """
#         try:
#             self.logger.info(f"Placing stop loss order for {symbol} at {stop_loss_price}")
#             stop_loss_order = await self.client.futures_create_order(
#                 symbol=symbol,
#                 side='BUY',  # For a short position, stop loss order is a buy order
#                 type='STOP_MARKET',
#                 stopPrice=stop_loss_price,
#                 quantity=quantity,
#                 closePosition=True
#             )
#             self.logger.info(f"Stop loss order placed for {symbol} at stop price {stop_loss_price}: {stop_loss_order}")
#             return stop_loss_order
#         except Exception as e:
#             self.logger.error(f"Error placing stop loss order for {symbol}: {str(e)}")
#             return None

#     async def place_take_profit_order(self, symbol, take_profit_price, quantity):
#         """
#         Places a take profit order for the specified symbol.

#         Parameters:
#             symbol: Trading symbol (e.g., 'BTCUSDT').
#             take_profit_price: Take profit price.
#             quantity: Quantity of the asset to trade.

#         Returns:
#             Take profit order response.
#         """
#         try:
#             self.logger.info(f"Placing take profit order for {symbol} at {take_profit_price}")
#             take_profit_order = await self.client.futures_create_order(
#                 symbol=symbol,
#                 side='SELL',  # For a long position, take profit order is a sell order
#                 type='TAKE_PROFIT_MARKET',
#                 stopPrice=take_profit_price,
#                 quantity=quantity,
#                 closePosition=True
#             )
#             self.logger.info(f"Take profit order placed for {symbol} at take profit price {take_profit_price}: {take_profit_order}")
#             return take_profit_order
#         except Exception as e:
#             self.logger.error(f"Error placing take profit order for {symbol}: {str(e)}")
#             return None

#     async def place_position_with_stop_loss_and_take_profit(self, symbol, current_price, amount_usdt, take_profit_percentage=0.05):
#         """
#         Places a position order and sets stop loss and take profit orders.

#         Parameters:
#             symbol: Trading symbol (e.g., 'BTCUSDT').
#             current_price: Current market price of the asset.
#             amount_usdt: Amount in USDT to use for the position.
#             take_profit_percentage: Percentage for the take profit price (default is 5%).

#         Returns:
#             Response of the stop loss and take profit orders.
#         """
#         try:
#             # Calculate position size
#             position_size = await self.calculate_position_size(symbol, current_price, amount_usdt)

#             # Place the market order
#             self.logger.info(f"Placing market order for {symbol} with size {position_size}")
#             order = await self.client.futures_create_order(
#                 symbol=symbol,
#                 side='BUY',  # Assuming it's a long position for simplicity
#                 type='MARKET',
#                 quantity=position_size
#             )

#             self.logger.info(f"Order placed for {symbol}: {order}")

#             # Calculate stop-loss price (5% from entry price)
#             stop_loss_price = Decimal(str(current_price)) * Decimal('0.95')

#             # Place the stop-loss order
#             stop_loss_order = await self.place_stop_loss_order(symbol, stop_loss_price, position_size)

#             # Calculate take-profit price (5% above entry price)
#             take_profit_price = Decimal(str(current_price)) * (Decimal('1.0') + Decimal(str(take_profit_percentage)))

#             # Place the take-profit order
#             take_profit_order = await self.place_take_profit_order(symbol, take_profit_price, position_size)

#             return {'stop_loss_order': stop_loss_order, 'take_profit_order': take_profit_order}

#         except Exception as e:
#             self.logger.error(f"Error placing position with stop loss and take profit for {symbol}: {e}")
#             return None

#     async def close(self):
#         """Close the Binance client connection."""
#         try:
#             await self.client.close_connection()
#             self.logger.info("Binance async client connection closed.")
#         except Exception as e:
#             self.logger.error(f"Error closing Binance client connection: {e}")
