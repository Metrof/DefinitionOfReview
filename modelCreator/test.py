import keras  # или `import tensorflow.keras as keras`

# 1. Загружаем из старого файла
model = keras.models.load_model("sentiment_model.h5")

# 2. Сохраняем в новом формате
model.save("keras_model.keras")