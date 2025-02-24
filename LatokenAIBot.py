import time
import asyncio
import json
import faiss
import numpy as np
import tiktoken
from aiogram import Bot, Dispatcher, types
from aiogram.types import Message
from aiogram.filters import CommandStart
from openai import OpenAI
from config import openai_api_key, telegram_token, ai_bot_prompt, starter_answer

# Подключаем OpenAI
client = OpenAI(api_key=openai_api_key)

# Загружаем FAISS
index = faiss.read_index("faiss_index.bin")
with open("faiss_texts.json", "r", encoding="utf-8") as f:
    texts = json.load(f)

# Храним историю чатов пользователей
user_histories = {}

# Функция для поиска в FAISS
def search_faiss(query, top_k=3):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    query_embedding = client.embeddings.create(input=query, model="text-embedding-ada-002").data[0].embedding
    query_embedding = np.array(query_embedding).astype("float32").reshape(1, -1)

    distances, indices = index.search(query_embedding, top_k)
    results = [texts[i] for i in indices[0] if i < len(texts)]
    return "\n".join(results)

# Подключаем Telegram-бота
bot = Bot(token=telegram_token)
dp = Dispatcher()

# Функция для запроса к OpenAI Assistant (Теперь с историей!)
async def ask_assistant(user_id, user_msg):
    # Загружаем историю чата пользователя
    if user_id not in user_histories:
        user_histories[user_id] = []

    # Ищем релевантный текст в FAISS
    context = search_faiss(user_msg)

    # Добавляем новый вопрос пользователя в историю
    user_histories[user_id].append({"role": "user", "content": user_msg})

    # Формируем полный контекст с историей
    messages = [
        {"role": "system", "content": ai_bot_prompt},
        {"role": "user", "content": f"Контекст: {context}"}
    ] + user_histories[user_id]

    # Отправляем запрос в GPT
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=messages
    )

    # Добавляем ответ бота в историю
    bot_response = response.choices[0].message.content
    user_histories[user_id].append({"role": "assistant", "content": bot_response})

    return bot_response

# Обработчик команды /start
@dp.message(CommandStart())
async def start_cmd(message: Message):
    await message.answer(starter_answer)

# Обработчик входящих сообщений
@dp.message()
async def handle_message(message: Message):
    user_id = message.from_user.id
    user_msg = message.text
    response = await ask_assistant(user_id, user_msg)
    await message.answer(response)

# Запуск бота
async def main():
    print("Бот запущен!")
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
