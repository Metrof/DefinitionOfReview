from fastapi import FastAPI, HTTPException
import keras
import pickle
from pydantic import BaseModel
import uvicorn
from keras.preprocessing.sequence import pad_sequences

app = FastAPI(title="Sentiment Prediction API")

try:
    model = keras.models.load_model("sentiment_model.keras")
except Exception as e:
    raise RuntimeError(f"Failed to load model: {e}")

try:
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
except Exception as e:
    raise RuntimeError(f"Failed to load tokenizer: {e}")

class PredictionInput(BaseModel):
    title: str
    text: str

@app.get("/")
async def root():
    return {"message": "Use POST /predict with JSON {\"title\": ..., \"text\": ...}"}

@app.get("/health")
async def health():
    return {"status": "OK"}

@app.post("/predict")
async def predict_sentiment(review: PredictionInput):
    if not isinstance(review.title, str) or not isinstance(review.text, str):
        raise HTTPException(status_code=400, detail="Invalid input: 'title' and 'text' must be strings.")

    try:
        full_text = f"{review.title} {review.text}"
        seq = tokenizer.texts_to_sequences([full_text])
        padded = pad_sequences(seq, maxlen=100, padding='post')
        prediction = model.predict(padded)[0][0]
        y_hat = int((prediction >= 0.5).astype(int))
        label = "Positive" if y_hat == 1 else "Negative"
    except Exception as ex:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {ex}")

    return {
        "label": label,
        "probability": float(prediction)
    }

host_rem = "0.0.0.0"
host_lock = "127.0.0.1"

if __name__ == "__main__":
    uvicorn.run(app, host=host_rem, port=8080)
