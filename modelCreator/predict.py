import keras  # или `import tensorflow.keras as keras`
import pickle
import pandas as pd
import matplotlib.pyplot as plt

model_name1 = "keras_model.keras"
model_name2 = "sentiment_model.keras"

# 1. Загружаем из старого файла
model = keras.models.load_model(f"../modelCreator/{model_name1}")

# title = "Awesome"
# text = "It's a great product"

title = "So bad"
text = "I dont like it"

test_texts = title + " " + text

with open("../tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# предобработка
seq = tokenizer.texts_to_sequences(test_texts)
padded = keras.preprocessing.sequence.pad_sequences(seq, maxlen=100, padding='post')

# predictions = model.predict(padded)[0][0]
#
# threshold = 0.5
#
# y_hat = (predictions >= threshold).astype(int)
#
# print("Pos" if y_hat == 1 else "Neg")

df_test = pd.read_csv("test.csv")

test_texts = (df_test.iloc[10000:10200, 1].astype(str) + " " + df_test.iloc[10000:10200, 2].astype(str)).values
test_labels = (df_test.iloc[10000:10200, :1].values - 1) # 0 или 1


sequences_test = tokenizer.texts_to_sequences(test_texts)
padded_test = keras.preprocessing.sequence.pad_sequences(sequences_test, padding='post', maxlen=100)
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