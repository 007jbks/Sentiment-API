import pandas as pd
import numpy as np
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Read CSV with correct encoding
df = pd.read_csv('D:/source/archive/train.csv', encoding='ISO-8859-1')

print("CSV read..")

# Fill missing values
df['text'] = df['text'].fillna("")
df['sentiment'] = df['sentiment'].fillna("neutral")

# Convert sentiment labels to numerical values
label_mapping = {"neutral": 0, "positive": 1, "negative": 2}
df['sentiment'] = df['sentiment'].map(label_mapping)

# Convert to NumPy arrays
X = df['text']
y = df['sentiment'].values  # Convert to NumPy array

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X_transformed = vectorizer.fit_transform(X).toarray()  # Convert to dense NumPy array

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)

# Build Model
print("Building Model...")
model = keras.Sequential([
    keras.Input(shape=(X_train.shape[1],)),
    layers.Dense(512, activation='relu'),
    layers.Dense(256, activation='relu'),
    layers.Dense(3, activation='softmax')  # 3 classes (neutral, positive, negative)
])
print("Model made..")

# Compile Model
model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    metrics=["accuracy"]
)

# Train Model
model.fit(X_train, y_train, batch_size=32, epochs=5, verbose=2)

# Evaluate Model
model.evaluate(X_test, y_test, batch_size=32, verbose=2)

import joblib

# Save the trained model
model.save("sentiment_model.h5")

# Save the vectorizer
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("Model and vectorizer saved successfully!")


text = input("Enter the text:")

def predict_sentiment(text):
    text_trans = vectorizer.transform([text]).toarray()  
    prediction = model.predict(text_trans)
    predicted_label = np.argmax(prediction)
    label_mapping_reverse = {0: "Neutral", 1: "Positive", 2: "Negative"}
    return label_mapping_reverse[predicted_label]


