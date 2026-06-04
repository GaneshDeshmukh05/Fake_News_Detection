import streamlit as st
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt

from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# Load Models
svm_model = pickle.load(open("model/fake_news_model.pkl","rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl","rb"))

tokenizer = DistilBertTokenizer.from_pretrained("distilbert_fake_news_model")

bert_model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert_fake_news_model"
)

# Page Title
st.title(" AI Fake News Detection Dashboard")

st.write("Detect whether a news article is **Fake or Real** using AI models.")

# Text Input
news = st.text_area("Enter News Text")

# Prediction Button
if st.button("Analyze News"):

    #SVM Prediction
    vec = vectorizer.transform([news])
    svm_pred = svm_model.predict(vec)[0]

    #DistilBERT Prediction
    inputs = tokenizer(news, return_tensors="pt", truncation=True, padding=True)

    outputs = bert_model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1).detach().numpy()[0]

    bert_pred = np.argmax(probs)

    confidence = probs[bert_pred] * 100

    # Display Results
    st.subheader("Prediction Result")

    if bert_pred == 0:
        st.error("Fake News Detected")
    else:
        st.success("Real News")

    st.write("Confidence:", round(confidence,2),"%")

    