import os
import asyncio
from playsound import playsound
from asyncio import Queue

# Define the base directory where sound files are stored
BASE_DIRECTORY = r"T:\Trading BOT Crypto - Profit\data\Sound_confirmations"

def get_sound_file_path(filename):
    # Construct the full path to the sound file
    sound_file_path = os.path.join(BASE_DIRECTORY, filename)
    return sound_file_path

async def play_sound(sound_file):
    sound_file_path = get_sound_file_path(sound_file)
    if os.path.exists(sound_file_path):
        playsound(sound_file_path)
    else:
        print(f"Sound file {sound_file} not found at {sound_file_path}")

async def sound_player(queue: Queue):
    while True:
        sound_file = await queue.get()
        await play_sound(sound_file)
        queue.task_done()

async def notify_sound(queue: Queue, sound_file: str):
    await queue.put(sound_file)

async def main():
    sound_queue = Queue()

    # Start the sound player
    asyncio.create_task(sound_player(sound_queue))

    # Example notifications
    await notify_sound(sound_queue, "Short position opened.mp3")
    await notify_sound(sound_queue, "Long position opened.mp3")
    await notify_sound(sound_queue, "Short signal confirmation.mp3")
    await notify_sound(sound_queue, "Long signal confirmation.mp3")
    await notify_sound(sound_queue, "Short position closed.mp3")
    await notify_sound(sound_queue, "Long position closed.mp3")

if __name__ == "__main__":
    asyncio.run(main())
