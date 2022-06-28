#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from pickle import dump
from pickle import load
import pickle
import matplotlib.pyplot as plt
import textblob
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import re


# In[2]:


import warnings
warnings.filterwarnings('ignore')


# In[3]:


st.title('Analysing User Review Ratings')
st.sidebar.header('Instructions')

st.sidebar.markdown("1.Review column's name should be **Review/Text**")
st.sidebar.markdown("2.Rating column's name should be **ratings**")
st.sidebar.markdown("3.Rating range should be 0to5")


# In[5]:


import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
p = PorterStemmer()
all_stopwords= stopwords.words('english')
all_stopwords.remove('not')

port = PorterStemmer()
# In[30]:


def text_cleaner(text):
    cleaned = re.sub('[^a-zA-Z]', " ", text)
    cleaned = cleaned.lower()
    cleaned = cleaned.split()
    cleaned = [port.stem(word) for word in cleaned if word not in stopwords.words("english")]
    cleaned = ' '.join(cleaned)
    return cleaned


# In[ ]:


st.title("Identifying Review's Rating")
st.header("Instructions")
st.markdown("1.Review column's name should be **review/text**")
st.markdown("2.Rating column's name should be **rating**")
st.markdown("3.Rating range should be 0to5")


# In[31]:


uploaded_file = st.file_uploader("Choose a File")

df = pd.read_csv("chrome_reviews.csv")
st.write(df)


# In[32]:


if st.button("Click for Results") :
    df["Text_cleaned"] = df["Text"].apply(lambda x: text_cleaner(str(x)))

    sid = SentimentIntensityAnalyzer()

    df["sentiments_vader"] = df["Text_cleaned"].apply(lambda x:sid.polarity_scores(x))
    df["Vader_Compound_Score"]  = df['sentiments_vader'].apply(lambda score_dict: score_dict['compound'])
    df["Result"] = df["Vader_Compound_Score"].apply(lambda c: 'positive' if c > 0 else ('negative' if c < 0 else 'neutral'))
    st.bar_chart(df.Result.value_counts())

    df_positive = df[(df.Result == "positive")]
    df_positive["Opinion_Positive"] = df_positive["Star"].apply(lambda star: "No Attention Needed" if star >= 3 else "Attention Needed")
    st.bar_chart(df_positive.Opinion_Positive.value_counts())

    data = df_positive

    st.download_button(
        label="Download data as CSV",
        data=data.to_csv().encode("utf-8"),
        file_name='data.csv',
        mime='text/csv',
    )

