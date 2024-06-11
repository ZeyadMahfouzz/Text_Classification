import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

data = pd.read_csv(r"text_categories.csv")
x = data['Text']
y = data['Label']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=0)
vectorizer = CountVectorizer().fit(x_train)
text_train_vectorized = vectorizer.transform(x_train)
text_test_vectorized = vectorizer.transform(x_test)
clfr = MultinomialNB()
clfr.fit(text_train_vectorized,y_train)
predicted = clfr.predict(text_test_vectorized)

st.title("Spam Classification App")
st.image(r"text_image.png",use_column_width=True)
st.text('Model Description: Naive Bayes Model, trained on text classification data')
st.text('to predict if the provided text falls under one of the following categories:')
st.text('Politics')
st.text('Sports')
st.text('Technology')
st.text('Entertainment')
st.text('Business')
        
text = st.text_input("Enter Text Here","Type Here...")
predict = st.button('Predict')

if predict:
    new_test_data = vectorizer.transform([text])
    predicted_label = clfr.predict(new_test_data)[0]
    if predicted_label == 0:
        prediction_text = "Politics"
        st.success(f"'{text}' is classified as {prediction_text}")
    elif predicted_label == 1:
        prediction_text = "Sport"
        st.success(f"'{text}' is classified as {prediction_text}")
    elif predicted_label == 2:
        prediction_text = "Technology"
        st.success(f"'{text}' is classified as {prediction_text}")
    elif predicted_label == 3:
        prediction_text = "Entertainment"
        st.success(f"'{text}' is classified as {prediction_text}")
    elif predicted_label == 4:
        prediction_text = "Business"
        st.success(f"'{text}' is classified as {prediction_text}")



