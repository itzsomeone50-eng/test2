import streamlit as st
import pickle

# Load pipeline (vectorizer + model already inside)
with open("model_class.pkl", "rb") as f:
    pipeline = pickle.load(f)

st.title("📌 Sentiment Analyzer")

user_text = st.text_input("Enter your text:")

if st.button("Check Sentiment"):
    if user_text:
        prediction = pipeline.predict([user_text])[0]
        res = {0: "Negative", 1: "Positive"}
        st.write(f"**Prediction (Sentiment of the Given text):** {res[prediction]}")
    else:
        st.warning("Please enter some text.")
