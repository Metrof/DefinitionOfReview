# Используем официальный Python образ
FROM python:3.11-slim

# Установка рабочих зависимостей
WORKDIR /app

# Копируем requirements (если есть)
COPY requirements.txt .

# Устанавливаем зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Копируем всё, кроме modelCreator
COPY . .

# Удаляем ненужную папку (на всякий случай, если попала)
RUN rm -rf modelCreator

# Открываем порт
EXPOSE 8080

# Запуск uvicorn
CMD ["uvicorn", "api_service:app", "--host", "0.0.0.0", "--port", "8080"]
