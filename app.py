import nltk
nltk.download('stopwords')
nltk.download('wordnet')
import streamlit as st
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

model = pickle.load(open("model.pkl","rb"))
vectorizer = pickle.load(open("vectorizer.pkl","rb"))

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

st.title("Restaurant Review Sentiment Analysis")

review = st.text_input("Enter Review")

if st.button("Predict"):

    clean = clean_text(review)
    vector = vectorizer.transform([clean])
    prediction = model.predict(vector)

    if prediction[0] == 1:
        st.success("Positive Review 🙂")
    else:
        st.error("Negative Review 🙁")