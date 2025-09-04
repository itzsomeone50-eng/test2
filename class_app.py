import streamlit as st
import pickle

# Load pre-trained sentiment model
with open("model_class.pkl", "rb") as file:
    model = pickle.load(file)

st.title("📌 Sentiment Analyzer")

# Text input
user_text = st.text_input("Enter your text:")

# Button
if st.button("Check Sentiment"):
    if user_text:
        prediction = model.predict([user_text])[0]
        st.write(f"**Prediction:** {prediction}")
    else:
        st.warning("Please enter some text.")
