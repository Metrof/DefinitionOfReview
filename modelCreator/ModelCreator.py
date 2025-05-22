import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import pickle

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)


df_train = pd.read_csv("train.csv")

train_texts = (df_train.iloc[:100, 1].astype(str) + " " + df_train.iloc[:100, 2].astype(str)).values
train_labels = (df_train.iloc[:100, :1].values - 1) # 0 или 1
# 0 - neg
# 1 - pos

# Токенизация
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
# Tokenizer	- Преобразует текст → числа
tokenizer.fit_on_texts(train_texts)
sequences = tokenizer.texts_to_sequences(train_texts)

# Padding
padded = pad_sequences(sequences, padding='post', maxlen=100)
# pad_sequences - Делает все входы одинаковой длины

model = Sequential([
        tf.keras.layers.Embedding(input_dim=10000, output_dim=16, input_length=100),
        # Embedding - Обучаемое представление слов
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
        # LSTM / Bidirectional - Обрабатывает последовательность
        Dense(128, activation = 'relu', name='L1'),
        Dropout(0.5),
        Dense(64, activation='relu', name='L2'),
        Dropout(0.5),
        Dense(1, activation = 'sigmoid', name='L3')
])

model.compile(
    loss = tf.keras.losses.BinaryCrossentropy(),
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics=['accuracy']
)


model.fit(padded, train_labels, epochs=10, validation_split=0.2)

df_test = pd.read_csv("test.csv")

test_texts = (df_test.iloc[:100, 1].astype(str) + " " + df_test.iloc[:100, 2].astype(str)).values
test_labels = (df_test.iloc[:100, :1].values - 1) # 0 или 1


sequences_test = tokenizer.texts_to_sequences(test_texts)
padded_test = pad_sequences(sequences_test, padding='post', maxlen=100)
# pad_sequences(...) — дополнить до нужной длины

predictions = model.predict(padded_test)

threshold = 0.5

y_hat = (predictions >= threshold).astype(int)


# Преобразуем метки к одному формату
true_labels = test_labels.reshape(-1)
pred_labels = y_hat.reshape(-1)

# Подсчёт TP / FP / FN / TN (опционально)
correct = (true_labels == pred_labels)
incorrect = ~correct

plt.figure(figsize=(6, 4))
plt.hist([true_labels[correct], true_labels[incorrect]],
         bins=[0, 1, 2], label=['Правильно', 'Ошибки'], rwidth=0.6)
plt.xticks([0, 1])
plt.xlabel("Класс (0 = neg, 1 = pos)")
plt.ylabel("Количество")
plt.title("Сравнение верных и неверных предсказаний")
plt.legend()
plt.grid(True)
plt.show()

model.save("sentiment_model.h5")
with open("../tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)
