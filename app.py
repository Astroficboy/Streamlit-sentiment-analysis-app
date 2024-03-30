import streamlit as st
from wordcloud import WordCloud as wc
import numpy as np
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
# nltk.download('vader_lexicon')

text = st.text_area(label="Drop your text here...")

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

# def dataframe(sentiment):
#     sentiment_scores = []
#     for values in sentiment.values():
#         sentiment_scores.append(values)
#     array = np.array(sentiment_scores)
#     return pd.DataFrame({"Negative": sentiment['neg'],
#                          "Neutral": sentiment['neu'],
#                          "Positive": sentiment['pos'],
#                          "Compound": sentiment['compound']}, index=[0])




# data = dataframe(sentiment)
# st.write(data.head())
# st.bar_chart(data)c

st.bar_chart(sentiment)

