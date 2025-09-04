import streamlit as st
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer

# Load pre-trained sentiment model
with open("model_class.pkl", "rb") as file:
    model = pickle.load(file)

st.title("📌 Sentiment Analyzer")

# Text input
user_text = st.text_input("Enter your text:")
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
# Button
if st.button("Check Sentiment"):
    if user_text:
        user_text = vectorizer.fit_transform(user_text)
        prediction = model.predict([user_text])[0]
        st.write(f"**Prediction:** {prediction}")
    else:
        st.warning("Please enter some text.")

