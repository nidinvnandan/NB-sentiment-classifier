import streamlit as st
from sentiment import naive_bayes as o
st.header('Sentiment Analysis')
input = st.text_area("Please enter the text whose category you would like to know", value="")
l=[]
l.append(input)
if st.button("Predict"):
    
    
    vec = o.vector.transform(l).toarray()
    k=str(list(o.naivebayes.predict(vec))[0]).replace('0', 'Negative').replace('1', 'positive')
    st.write('The Category to which the given text belongs is: ',k)