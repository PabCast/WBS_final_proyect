import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import re
from nltk.stem.porter import PorterStemmer

# Load Financial Dictionary and Tweets
def load_data():
    fin_master_df = pd.read_csv("C:\\Users\\49176\\OneDrive\\Desktop\\BC_final\\2_Financial_dict\\posneg_wordlisting_finance.csv")
    tweets_df = pd.read_csv("C:\\Users\\49176\\OneDrive\\Desktop\\BC_final\\1_StockTwits_clean\\AAPL_tweets_clean.csv")
    return fin_master_df, tweets_df

fin_master_df, tweets_df = load_data()

# Convert DataFrame to Dictionary for quick lookup
fin_master_dict = pd.Series(fin_master_df.Label.values, index=fin_master_df.Word).to_dict()

# Set weight for positive values based on the ratio of Negative to Positive labels in your dictionary
positive_weight = 2345 / 347  # Adjust this value based on your specific needs

def label_tweet_optimized(tweet):
    tweet_words = tweet.lower().split()
    positive_count = 0
    negative_count = 0
    
    for word in tweet_words:
        sentiment = fin_master_dict.get(word, None)
        if sentiment == 'Positive':
            positive_count += positive_weight
        elif sentiment == 'Negative':
            negative_count += 1

    if positive_count > negative_count:
        return 'Positive'
    elif negative_count > positive_count:
        return 'Negative'
    else:
        return 'Neutral'

tweets_df['Label'] = tweets_df['cleaned_body'].apply(label_tweet_optimized)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(tweets_df['cleaned_body'])

# Filtering out 'Neutral' tweets for binary classification
tweets_filtered_df = tweets_df[tweets_df['Label'] != 'Neutral']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X[tweets_filtered_df.index], 
    tweets_filtered_df['Label'], 
    test_size=0.2, 
    random_state=42)

# Logistic Regression Model
model = LogisticRegression(max_iter=5000, solver='saga')
model.fit(X_train, y_train)

def preprocess_tweet(text):
    # Lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove mentions and hashtag
    text = re.sub(r'@\w+|\#','', text)
    
    # Remove punctuations
    text = re.sub(r'[^\w\s]', '', text)
    
    # Stemming
    stemmer = PorterStemmer()
    text = ' '.join([stemmer.stem(word) for word in text.split()])
    
    return text

# Streamlit app
st.title("Tweet Sentiment Analysis")

user_input = st.text_area("Enter a tweet for sentiment analysis:")

if user_input:
    cleaned_tweet = preprocess_tweet(user_input)
    tweet_vectorized = vectorizer.transform([cleaned_tweet])
    prediction = model.predict(tweet_vectorized)
    st.write(f"The model's sentiment prediction for the new tweet is: **{prediction[0]}**")
