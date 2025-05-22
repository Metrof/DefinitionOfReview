import streamlit as st
import requests

# Настройка URL (замени при необходимости на внешний адрес)
API_URL = "https://sentiment-76136330633.europe-central2.run.app/predict"

st.set_page_config(page_title="Sentiment Checker", layout="centered")
st.title("🧠 Проверка настроения отзыва через API")

with st.form("sentiment_form"):
    title = st.text_input("Заголовок отзыва")
    text = st.text_area("Текст отзыва")
    submit = st.form_submit_button("Предсказать")

if submit:
    if not title.strip() or not text.strip():
        st.warning("Пожалуйста, заполните оба поля.")
    else:
        try:
            payload = {"title": title, "text": text}
            response = requests.post(API_URL, json=payload)
            if response.status_code == 200:
                result = response.json()
                st.markdown(f"### 💬 Предсказание: **{result['label']}**")
                st.markdown(f"Вероятность: `{result['probability']:.2f}`")
            else:
                st.error(f"Ошибка API: {response.status_code} — {response.text}")
        except requests.exceptions.RequestException as e:
            st.error(f"Не удалось подключиться к API: {e}")