# importing Dependencies
import pandas as pd
import numpy as np
import nltk
import re
#import spacy
#nlp = spacy.load("en_core_web_sm")
import nltk
from nltk.corpus import stopwords
from afinn import Afinn
afn = Afinn()
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score,classification_report
from sklearn.svm import SVC
import streamlit as st
import time


#function for text preprocessing
def get_cleanText(text):
    text = re.sub(r'@[A-Za-z0-9]+','',text)#removes username
    text = re.sub(r'#','',text)
    text = re.sub(r'RT[\s]+','',text)
    text = re.sub(r'https?:\/\/\S+','',text)# removes hyperlink
    text=  re.sub("[^A-Za-z" "]+"," ",text) #removes special characters
    return text


st.set_page_config(page_title='Sentiment_Analyzer', page_icon='🐣', layout="wide", initial_sidebar_state="expanded", menu_items=None)





#loading the data
df= pd.read_csv('data/modelbuilding.csv')



df= df.sample(frac=1)
X = df['lemmatized_text'].values.astype('U')
Y = df['sentiment']
vectorizer = TfidfVectorizer(stop_words='english',lowercase='True')
vectorizer.fit(X)
X_vectorized = vectorizer.transform(X)
smote = SMOTE(sampling_strategy='minority')
X_sm, Y_sm = smote.fit_resample(X_vectorized, Y)
X_sma, Y_sma = smote.fit_resample(X_sm,Y_sm)
#print(Y_sma.value_counts())

#X_train, X_test, Y_train, Y_test = train_test_split(X_sma,Y_sma,stratify=Y_sma,test_size=0.2)

####Model Building
#model = SVC(C=10,gamma=1,kernel='rbf')
#model.fit(X_train,Y_train)
#predictions = model.predict(X_test)
#st.write(classification_report(Y_test, predictions))
#st.write(accuracy_score(Y_test, predictions))


st.title("Sentment Analyzer..")

from PIL import Image
image = Image.open('data/sentiment2.jpg')
st.image(image,width=(1222))

st.markdown("\n")
st.markdown("\n")
text = st.text_area("Enter some text, a review given by a customer, a tweet, or "
             "anything else in the below text box and then click on the Button.",
                    height=175)
text1 = get_cleanText(text)
text2=[text1]
#text_vectorized =vectorizer.transform(text2)
#pred=model.predict(text_vectorized)
#st.write(pred)

def get_scores(text):
    score=afn.score(text)
    if score > 0:
        return 'positive'
    elif score < 0:
        return 'negative'
    else :
        return 'neutral'
col1, col2, col3 = st.columns([8,10,1])

with col1:
    st.write("")

with col2:
    st.button("Check Sentiment 🤔")
    with st.spinner('Wait for it...'):
        time.sleep(1)
output = get_scores(text1)
if output == 'positive':
    st.success("The text entered is positive.🤗")
    st.balloons()
elif output == 'negative':
    st.error("The text entered is Negative.😥")
else:
    st.info("The text entered is Nuetral.🙂")
    st.snow()

with col3:
    st.write("")




st.markdown("\n")
st.markdown("\n")
st.markdown("\n")
st.markdown("\n")
st.markdown("\n")
st.markdown("\n")
st.markdown("\n")
st.markdown("\n")
st.markdown("\n")
st.markdown("\n")
st.markdown("\n")
st.markdown("\n")
st.markdown("\n")
st.markdown("\n")
st.markdown("\n")
st.markdown("\n")
st.markdown("\n")
st.markdown("\n")
st.markdown("\n")
st.markdown("\n")
st.write(" ")
st.write(" ")
st.write(" ")
st.markdown("***")


#st.markdown("---")
st.markdown("- Developed by `SKY`.   ⇨[Ig](https://www.instagram.com/suraj452/),[github ](https://github.com/suraj4502), [Linkedin](https://www.linkedin.com/in/surajkumar-yadav-6ab2011a4/).")
#st.markdown("---")
