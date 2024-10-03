# def record_trade_log(symbol, entry_price, exit_price, profit):
#     # Create or append to trade log DataFrame
#     global trade_log
#     trade_log = trade_log.append({
#         'Symbol': symbol,
#         'Entry Price': entry_price,
#         'Exit Price': exit_price,
#         'Profit': profit
#     }, ignore_index=True)

#     # Save trade log to Excel file
#     trade_log.to_excel('trade_log.xlsx', index=False)
#     logging.info(f"Trade logged for {symbol} - Entry Price: {entry_price}, Exit Price: {exit_price}, Profit: {profit}")