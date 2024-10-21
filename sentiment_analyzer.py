# Import necessary libraries
import pandas as pd
import numpy as np
import re
import nltk
import torch

from datasets import load_dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Preprocessing Functions
stop_words = set(nltk.corpus.stopwords.words('english'))

def clean_text(text):
    # Remove URLs, mentions, hashtags, and special characters
    text = re.sub(r'http\S+|@\w+|#\w+', '', text)
    text = re.sub(r'[^A-Za-z\s$]', '', text)  # Keep dollar sign for stock symbols
    text = text.lower().strip()
    return text

def remove_stop_words(text):
    tokens = nltk.word_tokenize(text)
    filtered = [word for word in tokens if word not in stop_words]
    return ' '.join(filtered)

def preprocess_text(text):
    text = clean_text(text)
    text = remove_stop_words(text)
    return text

# Load the sentiment training dataset
print("Loading sentiment training dataset...")
sentiment_dataset = load_dataset('sentiment140', trust_remote_code=True) 
sentiment_df = sentiment_dataset['train'].to_pandas()

# Check the columns
print("Columns in the DataFrame:", sentiment_df.columns)



# Map sentiment labels
sentiment_label_mapping = {0: 'negative',2: 'neutral', 4: 'positive'}
sentiment_df['label'] = sentiment_df['sentiment'].map(sentiment_label_mapping)

# Ensure 'label' column was created
if 'label' not in sentiment_df.columns:
    print("Error: 'label' column could not be created.")
    exit()

# Preprocess the sentiment data
print("Preprocessing sentiment training data...")
sentiment_df['clean_text'] = sentiment_df['text'].apply(preprocess_text)

# Encode labels
le_sentiment = LabelEncoder()
sentiment_df['label_encoded'] = le_sentiment.fit_transform(sentiment_df['label'])

# Split data into features and labels
X_sentiment = sentiment_df['clean_text']
y_sentiment = sentiment_df['label_encoded']

# Vectorize text data using TF-IDF
print("Vectorizing text data...")
vectorizer_senti = TfidfVectorizer(max_features=5000)
X_sentiment_tfidf = vectorizer_senti.fit_transform(X_sentiment)

# Train the sentiment analysis model
print("Training sentiment analysis model...")
lr_model = LogisticRegression(max_iter=200, multi_class='ovr')
lr_model.fit(X_sentiment_tfidf, y_sentiment)

# Save the trained model and encoders
import joblib
joblib.dump(lr_model, 'sentiment_lr_model.joblib')
joblib.dump(vectorizer_senti, 'tfidf_vectorizer.joblib')
joblib.dump(le_sentiment, 'label_encoder.joblib')

# Function to test the sentiment model with a sample text
def test_sentiment_model(sample_text):
    # Preprocess the text
    preprocessed_text = preprocess_text(sample_text)
    
    # Transform the text using the trained TF-IDF vectorizer
    text_tfidf = vectorizer_senti.transform([preprocessed_text])
    
    # Predict the sentiment using the trained model
    prediction = lr_model.predict(text_tfidf)
    
    # Convert the numerical prediction back to the original label
    predicted_sentiment = le_sentiment.inverse_transform(prediction)
    
    # Print the result
    print(f"Sample Text: {sample_text}")
    print(f"Predicted Sentiment: {predicted_sentiment[0]}")

# Examples to test
test_sentiment_model("I am so happy")
test_sentiment_model("This is terrible and I am very sad")
test_sentiment_model("It's an average day with nothing special")
test_sentiment_model("What the tuna is going on")
test_sentiment_model("This is going to be the worst day ever")
test_sentiment_model("My dog died")
test_sentiment_model("I won a million bucks")

# Load the trained model and encoders
print("Loading trained sentiment model...")
lr_model = joblib.load('sentiment_lr_model.joblib')
vectorizer_senti = joblib.load('tfidf_vectorizer.joblib')
le_sentiment = joblib.load('label_encoder.joblib')

# Load the stock dataset
print("Loading stock dataset...")
stock_dataset = load_dataset('suchkow/twitter-sentiment-stock', trust_remote_code=True)
stock_df = stock_dataset['train'].to_pandas()

# Set the text column to 'Tweet'
text_column = 'Tweet'

# Verify that 'Tweet' is in the columns
print("Columns in stock_df:", stock_df.columns)

if text_column in stock_df.columns:
    print(f"Using '{text_column}' as the text column.")
    # Preprocess the stock data
    stock_df['clean_text'] = stock_df[text_column].apply(preprocess_text)
else:
    print(f"Error: Column '{text_column}' not found in stock_df.")
    print("Available columns:", stock_df.columns)
    exit()

# Extract stock symbols from tweets
def extract_tickers(text):
    tickers = re.findall(r'\$[A-Za-z]+', text)
    tickers = [ticker.strip('$').upper() for ticker in tickers]
    return tickers

stock_df['tickers'] = stock_df[text_column].apply(extract_tickers)

# Function to get sentiment for a specific stock
def get_stock_sentiment(stock_symbol):
    stock_symbol_upper = stock_symbol.upper()
    filtered_df = stock_df[stock_df['tickers'].apply(lambda x: stock_symbol_upper in x)]
    
    if filtered_df.empty:
        print(f"No tweets found for stock symbol {stock_symbol_upper} in the dataset.")
        return None
    
    print(f"Number of tweets found for {stock_symbol_upper}: {len(filtered_df)}")
    
    # Get the clean_text of filtered tweets
    tweets = filtered_df['clean_text'].tolist()
    
    # Vectorize tweets using the trained vectorizer
    tweets_tfidf = vectorizer_senti.transform(tweets)
    
    # Predict sentiment using the trained model
    predictions = lr_model.predict(tweets_tfidf)
    sentiments = le_sentiment.inverse_transform(predictions)
    
    # Create a DataFrame with the original tweets and predicted sentiments
    sentiment_results_df = pd.DataFrame({
        'Date': filtered_df['Date'].tolist(),
        'Tweet': filtered_df[text_column].tolist(),
        'Predicted Sentiment': sentiments
    })
    
    # Aggregate sentiment counts
    sentiment_counts = sentiment_results_df['Predicted Sentiment'].value_counts()
    total = sentiment_counts.sum()
    positive = sentiment_counts.get('positive', 0)
    neutral = sentiment_counts.get('neutral', 0)
    negative = sentiment_counts.get('negative', 0)
    
    print(f"Sentiment analysis for {stock_symbol_upper}:")
    print(f"Positive tweets: {positive} ({(positive/total)*100:.2f}%)")
    print(f"Neutral tweets: {neutral} ({(neutral/total)*100:.2f}%)")
    print(f"Negative tweets: {negative} ({(negative/total)*100:.2f}%)")
    
    # **Print the top 100 tweets and their predictions**
    print("\nTop 100 Tweets and their Predicted Sentiments:")
    top_100 = sentiment_results_df.head(100)
    for index, row in top_100.iterrows():
        print(f"Date: {row['Date']}")
        print(f"Tweet: {row['Tweet']}")
        print(f"Predicted Sentiment: {row['Predicted Sentiment']}\n")
    
    # Visualize the sentiment distribution
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    sns.countplot(x='Predicted Sentiment', data=sentiment_results_df)
    plt.title(f"Sentiment Distribution for {stock_symbol_upper}")
    plt.show()
    
    return sentiment_results_df

# Example usage
if __name__ == '__main__':
    stock_symbol = input("Enter a stock symbol (e.g., AAPL): ").upper()
    sentiment_df = get_stock_sentiment(stock_symbol)
