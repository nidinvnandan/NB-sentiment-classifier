import pandas as pd# Read our dataset using read_csv()
import pickle as pk
import streamlit as st
class naive_bayes:
    review = pd.read_csv('reviews (1).csv')
    review = review.rename(columns = {'text': 'review'}, inplace = False)
    review.head()
    from sklearn.model_selection import train_test_split
    X = review.review
    y = review.polarity
    #split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.6, random_state = 1)
    from sklearn.feature_extraction.text import CountVectorizer
    vector = CountVectorizer(stop_words = 'english',lowercase=False)
    # fit the vectorizer on the training data
    vector.fit(X_train)
    print(vector.vocabulary_)
    X_transformed = vector.transform(X_train)
    X_transformed.toarray()
    # for test data
    X_test_transformed = vector.transform(X_test)
    from sklearn.naive_bayes import MultinomialNB
    naivebayes = MultinomialNB()
    naivebayes.fit(X_transformed, y_train)
