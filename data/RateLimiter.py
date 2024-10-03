import asyncio
import time

# Constants for rate limiting
MAX_REQUESTS = 100  # Maximum number of requests
TIME_WINDOW = 300   # Time window in seconds (5 minutes)
REQUEST_DELAY = TIME_WINDOW / MAX_REQUESTS  # Delay between requests

class RateLimiter:
    def __init__(self, max_requests, time_window):
        self.semaphore = asyncio.Semaphore(max_requests)
        self.time_window = time_window
        self.request_times = []

    async def __aenter__(self):
        await self.semaphore.acquire()
        now = time.monotonic()

        # Remove outdated request timestamps
        self.request_times = [t for t in self.request_times if now - t < self.time_window]

        # If the number of requests is at max, calculate the delay
        if len(self.request_times) >= self.semaphore._value:
            wait_time = self.time_window - (now - self.request_times[0])
            if wait_time > 0:
                await asyncio.sleep(wait_time)

        self.request_times.append(now)

    async def __aexit__(self, exc_type, exc_value, traceback):
        self.semaphore.release()

# Create an instance of RateLimiter with defined constants
rate_limiter = RateLimiter(MAX_REQUESTS, TIME_WINDOW)
