import logging
import time
import asyncio
from datetime import datetime
from logger_config import get_logger

class RequestLogger:
    def __init__(self, log_file='request_logs.txt'):
        self.log_file = log_file
        self.request_count = 0
        self.request_summary = {}  # Store counts by request type and endpoint
        self.start_time = time.time()

    def log_request(self, method, endpoint):
        self.request_count += 1
        if method not in self.request_summary:
            self.request_summary[method] = {}
        
        if endpoint not in self.request_summary[method]:
            self.request_summary[method][endpoint] = 0
        
        self.request_summary[method][endpoint] += 1

        self.save_to_file()

    def save_to_file(self):
        current_time = time.time()
        time_diff = (current_time - self.start_time) / 3600  # Time in hours

        if time_diff >= 1:
            # Reset the logger every hour and save the current log to file
            with open(self.log_file, 'a') as f:
                log_data = f"Total requests in the past hour: {self.request_count}\n"
                log_data += "Request summary:\n"
                for method, endpoints in self.request_summary.items():
                    for endpoint, count in endpoints.items():
                        log_data += f"{method} {endpoint}: {count}\n"
                log_data += "\n"
                f.write(log_data)
            
            # Reset count and summary after logging
            self.request_count = 0
            self.request_summary = {}
            self.start_time = time.time()

    # async def log_and_execute(logger, client, method, endpoint, **kwargs):
    #     logger.log_request(method.upper(), endpoint)

    #     if endpoint == 'fapi/v2/balance':
    #         if method.upper() == 'GET':
    #             response = await client.futures_account_balance()
    #         else:
    #             raise ValueError("Unsupported request method for this endpoint")

    #     # Add more mappings for other endpoints and methods as needed
    #     else:
    #         raise ValueError("Unknown endpoint or unsupported method")

    #     return response

