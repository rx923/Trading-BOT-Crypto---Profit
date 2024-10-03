# import logging
# from binance.client import Client


# # def calculate_position_size(symbol, entry_price):
# #     try:
# #         # Example function to fetch account balance from exchange (replace with actual implementation)
# #         account_balance = client.futures_account_balance()  # Fetch account balance in USDT from exchange
# #         max_position_size_pct = 0.3  # Maximum 30% of account balance for position size
# #         max_trade_size = account_balance * max_position_size_pct
        
# #         # Ensure position size does not exceed max_trade_size
# #         position_size = min(max_trade_size / entry_price, max_trade_size)
        
# #         return position_size
    
# #     except Exception as e:
# #         logging.error(f"Error calculating position size for {symbol}: {e}")
# #         return None
    

# def calculate_stop_loss(symbol, entry_price):
#     try:
#         # Fetch current market price from Binance API
#         current_price = fetch_current_price(symbol)
        
#         # Log the current price
#         logging.info(f"Current price for {symbol}: {current_price}")
        
#         # Example: Calculate stop loss as 5% below the entry price (adjust as per your risk management)
#         max_stop_loss_pct = 0.05  # Maximum 5% of entry price for stop loss
#         stop_loss_price = entry_price * (1 - max_stop_loss_pct)
        
#         return stop_loss_price
    
#     except Exception as e:
#         logging.error(f"Error calculating stop loss for {symbol}: {e}")
#         return None