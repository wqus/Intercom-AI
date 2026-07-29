import asyncio
from aiogram import Bot, Dispatcher, F
from aiogram.types import Message

TOKEN = "8474418916:AAFOpwpmxvQBJhj2F9OC5A_bOXuY9r2k6XM"

bot = Bot(TOKEN)
dp = Dispatcher()


@dp.message()
async def get_chat_id(message: Message):
    print("Chat ID:", message.chat.id)
    print("Chat title:", message.chat.title)
    print("Chat type:", message.chat.type)


async def main():
    print("Started")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())