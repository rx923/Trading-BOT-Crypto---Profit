# from data.BinanceManagerHandler import (api_key, api_secret)
# from binance.enums import *
# from binance.client import Client
# import numpy as np
# import time 
# client = Client(api_key, api_secret)



# BINANCE_BASE_URL = 'https://api4.binance.com'

# # Function to monitor price fluctuations of the traded coin pair
# def monitor_price_fluctuations(symbol):
#     historical_prices = []
#     klines = client.get_historical_klines(symbol, Client.KLINE_INTERVAL_1DAY)
#     for kline in klines:
#         historical_prices.append(float(kline[4]))

#     fourier = np.fft.fft(historical_prices)
#     frequencies = np.fft.fftfreq(len(historical_prices))

#     dominant_frequencies = np.argsort(np.abs(fourier))[::-1]
#     dominant_frequency = frequencies[dominant_frequencies[0]]

#     predicted_price_level = np.mean(historical_prices) + np.sin(2 * np.pi * dominant_frequency) * np.max(historical_prices)

#     print(f"Predicted price level for the next 2 hours: {predicted_price_level}")