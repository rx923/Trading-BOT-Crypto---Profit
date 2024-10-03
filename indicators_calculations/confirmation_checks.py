# import asyncio
# import logging
# import pandas as pd
# from datetime import datetime, timedelta
# from data.Fetching_and_calculating_moving_averages_and_TEMA import MovingAverageTrader


# class ConfirmationChecks:
#     def __init__(self, symbol, confirmation_intervals, confirmation_checks, moving_average_trader):
#         """
#         Initialize the ConfirmationChecks object.

#         Parameters:
#         - confirmation_intervals (list of int): Time intervals (in seconds) for each confirmation check stage.
#         - confirmation_checks (list of int): Number of successful checks required for each stage.
#         - moving_average_trader (MovingAverageTrader): An instance of MovingAverageTrader for price checks.
#         """
#         self.confirmation_intervals = confirmation_intervals
#         self.confirmation_checks = confirmation_checks
#         # self.moving_average_trader = MovingAverageTrader(self, symbol, window_9 = 9, window_30 = 30)
#         self.symbol = symbol

#     async def run_confirmation_checks(self, symbol: str, df: pd.DataFrame, check_func):
#         """
#         Run a series of confirmation checks on a given symbol.

#         Parameters:
#         - symbol (str): The trading symbol (e.g., 'ETHUSDT').
#         - df (pd.DataFrame): DataFrame for candlestick price checks.
#         - check_func (function): An asynchronous function that performs the check and returns a boolean.

#         Returns:
#         - bool: True if all confirmation checks pass, otherwise False.
#         """
#         try:
#             first_check = await self.first_confirmation_loop(symbol, df, check_func)
#             if first_check:
#                 second_check = await self.second_confirmation_loop(symbol, df, check_func)
#                 if second_check:
#                     final_check = await self.final_confirmation_check(check_func)
#                     return final_check
#         except Exception as e:
#             logging.error(f"An unexpected error occurred while processing symbol {symbol}: {str(e)}")
        
#         return False



#     async def first_confirmation_loop(self, symbol, df, check_func, direction="short"):
#         """
#         Perform the first stage of confirmation checks in a loop.

#         Parameters:
#         - symbol (str): The trading symbol.
#         - df (pd.DataFrame): DataFrame for candlestick price checks.
#         - check_func (function): An asynchronous function that performs the check and returns a boolean.
#         - direction (str): Direction of the signal, default is "short".

#         Returns:
#         - bool: True if the first confirmation stage is successful, otherwise False.
#         """
#         consecutive_checks_1 = 0
#         start_time = datetime.now()
#         edge_case_triggered = False

#         while consecutive_checks_1 < self.confirmation_checks[0]:
#             try:
#                 if await check_func():
#                     consecutive_checks_1 += 1
#                     logging.info(f"Symbol {symbol}: TEMA confirmation for {direction}. First confirmation check {consecutive_checks_1}/{self.confirmation_checks[0]}")
#                 else:
#                     consecutive_checks_1 = 0
#                     logging.info(f"Symbol {symbol}: First {direction} confirmation condition failed, resetting checks.")

#                 elapsed_time = (datetime.now() - start_time).total_seconds()
#                 logging.debug(f"Elapsed time for edge case check: {elapsed_time} seconds")

#                 if elapsed_time > 600 and not edge_case_triggered:
#                     logging.info(f"Symbol {symbol}: Edge case triggered. TEMA in {direction} condition for more than 10 minutes.")
#                     edge_case_triggered = True
#                     end_edge_case_time = datetime.now() + timedelta(minutes=5)

#                     while datetime.now() < end_edge_case_time:
#                         try:
#                             if await check_func() and await MovingAverageTrader.check_candlestick_prices(symbol, df):
#                                 logging.info(f"Symbol {symbol}: Candlestick price condition passed for {direction}.")
#                                 break
#                             else:
#                                 logging.info(f"Symbol {symbol}: Conditions no longer valid during edge case monitoring for {direction}.")
#                                 await asyncio.sleep(0.5)
#                                 break
#                         except Exception as e:
#                             logging.error(f"Error during edge case monitoring: {str(e)}")
#                             break

#                 await asyncio.sleep(self.confirmation_intervals[0])
#             except Exception as e:
#                 logging.error(f"An error occurred during first confirmation check for symbol {symbol}: {str(e)}")
#                 await asyncio.sleep(2)

#         return consecutive_checks_1 >= self.confirmation_checks[0]

#     async def second_confirmation_loop(self, symbol, check_func, direction="short"):
#         """
#         Perform the second stage of confirmation checks in a loop.

#         Parameters:
#         - symbol (str): The trading symbol.
#         - check_func (function): An asynchronous function that performs the check and returns a boolean.
#         - direction (str): Direction of the signal, default is "short".

#         Returns:
#         - bool: True if the second confirmation stage is successful, otherwise False.
#         """
#         consecutive_checks_2 = 0

#         while consecutive_checks_2 < self.confirmation_checks[1]:
#             try:
#                 if await check_func():
#                     consecutive_checks_2 += 1
#                     logging.info(f"Symbol {symbol}: TEMA confirmation for {direction}. Second confirmation check {consecutive_checks_2}/{self.confirmation_checks[1]}")
#                 else:
#                     consecutive_checks_2 = 0
#                     logging.info(f"Symbol {symbol}: Second {direction} confirmation condition failed, resetting checks.")

#                 await asyncio.sleep(self.confirmation_intervals[1])
#             except Exception as e:
#                 logging.error(f"An error occurred during second confirmation check for symbol {symbol}: {str(e)}")
#                 await asyncio.sleep(1)

#         return consecutive_checks_2 >= self.confirmation_checks[1]

#     async def final_confirmation_check(self, check_func):
#         """
#         Perform the final confirmation check.

#         Parameters:
#         - check_func (function): An asynchronous function that performs the check and returns a boolean.

#         Returns:
#         - bool: Result of the final confirmation check.
#         """
#         await asyncio.sleep(self.confirmation_intervals[2])
#         try:
#             result = await check_func()
#             return result
#         except Exception as e:
#             logging.error(f"An error occurred during final confirmation check: {str(e)}")
#             return False
