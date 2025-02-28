from fastapi import FastAPI
from typing import Union
import joblib

app = FastAPI()

import pandas as pd
import numpy as np
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow import keras

model = tf.keras.models.load_model('sentiment_model.h5')
vectorizer = joblib.load("tfidf_vectorizer.pkl")


def predict_sentiment(text):
    text_trans = vectorizer.transform([text]).toarray()  
    prediction = model.predict(text_trans)
    predicted_label = np.argmax(prediction)
    label_mapping_reverse = {0: "Neutral", 1: "Positive", 2: "Negative"}
    return label_mapping_reverse[predicted_label]

from pydantic import BaseModel

class TextInput(BaseModel):
    text: str

@app.post("/text")
def give_sentiment(input_data: TextInput):
    sent = predict_sentiment(input_data.text)
    return {"Sentiment": sent}
