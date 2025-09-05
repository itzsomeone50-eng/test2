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

def predict_sentiment(texts, model, vectorizer):
    """Predicts sentiment (0 or 1) for new input texts"""
    if isinstance(texts, str):
        texts = [texts]
    
    text_tfidf = vectorizer.fit_transform(texts)
    predictions = model.predict(text_tfidf)
    
    return dict(zip(texts, predictions))
# Button
if st.button("Check Sentiment"):
    predictions = predict_sentiment("it was good movie", model, vectorizer)
    res = {0:"Negative",1:"Positive"}
    p = int(list(predictions.values())[0])

    f_r = res[p]

    st.write(f"**Prediction(Sentiment of the Given text):** {f_r}")
else:
    st.warning("Please enter some text.")


