import os
import pandas as pd
from aiogram import Bot, Dispatcher, types
from aiogram.types import ContentType, InputFile
from aiogram.filters import Command
import asyncio
from aiogram import F
from aiogram.types import FSInputFile
import data_processing
from ourmodel import SIModel

# Введите токен вашего бота
TOKEN = '7982846313:AAG_ZgtSWpyDcY062JWNUfnNokZ1jGeMLZE'


# Инициализация бота и диспетчера
bot = Bot(token=TOKEN)
dp = Dispatcher()
model = SIModel()
# Хэндлер для команды /start
@dp.message(Command("start"))
async def start(message: types.Message):
    await message.answer("Привет! Я бот для определения активности лекарств. Пришлите мне CSV файл, который содержит колонки: Strain, Cell, DOI, Smiles, и я предскажу биоактивность.")

@dp.message(F.document.mime_type == 'text/csv')
async def handle_document(message: types.Message):
    document = message.document
    file_info = await bot.get_file(document.file_id)  # Получаем информацию о файле
    file_path = f"downloads/{document.file_id}.csv"
    
    # Скачиваем файл
    await bot.download_file(file_info.file_path, destination=file_path)

    try:
        # Обработка файла CSV с помощью pandas
        df = pd.read_csv(file_path)

        prediction = model.predict(data_processing.process(df))
        df = df.join(prediction)
        # Сохраняем новый CSV файл
        output_file_path = f"downloads/output.csv"
        df.to_csv(output_file_path, index=False)
        
        # Отправляем обратно пользователю новый CSV файл
        output_file = FSInputFile(output_file_path)
        await message.answer_document(output_file)

        # Удаляем временные файлы после отправки
        os.remove(file_path)
        os.remove(output_file_path)

    except Exception as e:
        await message.reply(f"Ошибка при обработке файла: {e}")

# Основная функция для запуска бота
async def main():
    # Запускаем бота
    await dp.start_polling(bot)

if __name__ == '__main__':
    # Создаем папку для временных файлов, если ее нет
    if not os.path.exists('downloads'):
        os.makedirs('downloads')

    # Запуск бота с помощью asyncio
    asyncio.run(main())
