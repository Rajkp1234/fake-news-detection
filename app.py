import streamlit as st
import pandas as pd
import numpy as np
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

st.title("📰 Fake News Detection")

# Load Dataset
df = pd.read_csv("news.csv")   # <-- apni dataset file ka naam yahan likhna

def wordopt(text):
    text = text.lower()
    text = re.sub(r'\\W', ' ', text)
    return text

df["text"] = df["text"].apply(wordopt)

x = df["text"]
y = df["label"]

vectorization = TfidfVectorizer()
xv = vectorization.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(
    xv, y, test_size=0.25, random_state=42
)

lr_model = LogisticRegression()
lr_model.fit(x_train, y_train)

news = st.text_area("Enter News Article")

if st.button("Predict"):

    news = wordopt(news)

    vectorized_input = vectorization.transform([news])

    prediction = lr_model.predict(vectorized_input)

    if prediction[0] == 0:
        st.error("🚨 Fake News")
    else:
        st.success("✅ Real News")
