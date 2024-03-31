import streamlit as st
from wordcloud import WordCloud as wc
import numpy as np
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
nltk.download('vader_lexicon')

st.title("Get a sentiment analysis")
st.write("Size of the text does not matter.")

text = st.text_area(label="...", placeholder="Drop your text here...")

st.divider()

if text:
    # Generate word cloud
    wordcloud = wc().generate(text)
    
    # Convert word cloud image to NumPy array
    wordcloud_array = np.array(wordcloud)
    
    # Display word cloud image using Streamlit
    st.image(wordcloud_array, use_column_width=True)
    
analyser = SentimentIntensityAnalyzer()

def get_sentiment(txt):
    scores = analyser.polarity_scores(txt)
    return scores

sentiment = get_sentiment(text)

st.divider()

st.bar_chart(sentiment)

