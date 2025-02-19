import streamlit as st
from preprocess_data import PreprocessData
from network import nn_train
import torch



ppd = PreprocessData()
net = nn_train(file = "../data/processed_data.csv")
net.load_data()


st.title("Sentiment Analysis App")
st.subheader("Sentiment analysis using a neural network model")

text = st.text_input("### Enter a text to analyze:")

if text.strip() == "":
    st.markdown(f"""
        <p style="font-size:18px;" > Nothing entered. Please enter some text.</p>
    """, unsafe_allow_html=True)
    
else:
    text = ppd.preprocess_text(text)
    prediction = net.IO(text, "../data/model.pth")
    st.markdown(f"""
    <p style="font-size:18px;" >The sentiment of the text is {prediction}.</p>
    """, unsafe_allow_html=True)