import json
import faiss
import numpy as np
import tiktoken
import asyncio
from openai import AsyncOpenAI
from config import openai_api_key, my_json_file

# Подключаем OpenAI (асинхронный клиент)
client = AsyncOpenAI(api_key=openai_api_key)

# Загружаем JSON-файл
with open(my_json_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# Функция разбиения текста на чанки (500 токенов)
def split_text(text, chunk_size=500):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)

    # Преобразуем токены обратно в текст (правильный способ)
    chunks = [" ".join(tokenizer.decode(tokens[i:i+chunk_size]).split()) for i in range(0, len(tokens), chunk_size)]
    return chunks

# Подготовка данных
texts = []
index = faiss.IndexFlatL2(1536)  # Векторный индекс
embeddings = []

# Асинхронное получение эмбеддингов
async def get_embedding(text):
    response = await client.embeddings.create(input=text, model="text-embedding-ada-002")
    return response.data[0].embedding

async def process_data():
    tasks = []
    for key, value in data.items():
        chunks = split_text(str(value))
        for chunk in chunks:
            texts.append(chunk)
            tasks.append(get_embedding(chunk))  # Добавляем в список задач

    print(f"📌 Отправляем {len(tasks)} запросов на эмбеддинги...")
    results = await asyncio.gather(*tasks)  # Запускаем все запросы параллельно

    # Конвертируем в numpy и загружаем в FAISS
    embeddings_array = np.array(results).astype("float32")
    index.add(embeddings_array)

    # Сохраняем FAISS индекс
    faiss.write_index(index, "faiss_index.bin")

    # Сохраняем тексты
    with open("faiss_texts.json", "w", encoding="utf-8") as f:
        json.dump(texts, f, ensure_ascii=False, indent=4)

    print("Векторная база FAISS создана и сохранена")

# Запускаем асинхронную обработку
asyncio.run(process_data())
