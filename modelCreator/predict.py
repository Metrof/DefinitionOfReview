import keras  # или `import tensorflow.keras as keras`
import tensorflow as tf
import pickle

# 1. Загружаем из старого файла
model = keras.models.load_model("../keras_model.keras")

# title = "Awesome"
# text = "It's a great product"

title = "So bad"
text = "I dont like it"

test_texts = title + " " + text

with open("../tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# предобработка
seq = tokenizer.texts_to_sequences(test_texts)
padded = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=100, padding='post')

predictions = model.predict(padded)[0][0]

threshold = 0.5

y_hat = (predictions >= threshold).astype(int)

print("Pos" if y_hat == 1 else "Neg")