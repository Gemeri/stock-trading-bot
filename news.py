import alpaca_trade_api as tradeapi
import csv
from datetime import datetime, timedelta, timezone
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os
from dotenv import load_dotenv

# ==============================
# User configuration parameters
# ==============================

load_dotenv()

API_KEY = os.getenv("ALPACA_API_KEY", "")
SECRET_KEY = os.getenv("ALPACA_API_SECRET", "")
# Use your paper account endpoint (or live, if appropriate)
BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

# The stock symbol for which you want to fetch news articles (e.g., AAPL)
SYMBOL = os.getenv("TICKERS", "TSLA").split(",")

# Total number of days (back from today) for which to retrieve articles
NUM_DAYS = 1650

# How many articles per day to fetch (default is 1; change as needed, e.g., 5)
ARTICLES_PER_DAY = 2

# Output CSV file name
CSV_FILENAME = "alpaca_news_articles.csv"

# ==============================
# Date Setup & Alpaca API Initialization
# ==============================

# Create a timezone-aware "today" in UTC
today = datetime.now(timezone.utc)
# Set the start date to today minus NUM_DAYS
start_date = today - timedelta(days=NUM_DAYS)

# Initialize the Alpaca REST API client (v2)
api = tradeapi.REST(API_KEY, SECRET_KEY, BASE_URL, api_version="v2")

# ==============================
# Sentiment Analysis Model Setup
# ==============================

model_name = "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model(**inputs)
    sentiment_class = outputs.logits.argmax(dim=1).item()
    sentiment_mapping = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    predicted_sentiment = sentiment_mapping.get(sentiment_class, 'Unknown')
    # Get the confidence scores from the softmax of logits
    confidence_scores = outputs.logits.softmax(dim=1)[0].tolist()
    return predicted_sentiment, confidence_scores

# ==============================
# CSV Setup
# ==============================

# We will save only the following features: Headline, Summary, CreatedAt, UpdatedAt,
# plus sentiment information: Sentiment, Confidence_Negative, Confidence_Neutral, Confidence_Positive.
fieldnames = [
    "Headline", 
    "Summary", 
    "CreatedAt", 
    "UpdatedAt", 
    "Sentiment", 
    "Confidence_Negative", 
    "Confidence_Neutral", 
    "Confidence_Positive"
]

with open(CSV_FILENAME, mode="w", newline="", encoding="utf-8") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    # ==============================
    # Retrieve Articles for Each Day & Process
    # ==============================
    # Use NUM_DAYS+1 so that if today should be included, it will be.
    for day_offset in range(NUM_DAYS + 1):
        current_day = start_date + timedelta(days=day_offset)
        # Do not process dates in the future
        if current_day > today:
            print(f"Reached today's date: {today.date()}, stopping further requests.")
            break
        
        next_day = current_day + timedelta(days=1)
        # Format dates as ISO8601 with a trailing 'Z' to indicate UTC
        start_str = current_day.strftime("%Y-%m-%dT%H:%M:%SZ")
        end_str   = next_day.strftime("%Y-%m-%dT%H:%M:%SZ")
        
        try:
            # Fetch news articles published in the current 24-hour window for the specified symbol.
            articles = api.get_news(SYMBOL, start=start_str, end=end_str)
            
            if articles:
                # Limit the number of articles processed per day to ARTICLES_PER_DAY.
                count = min(len(articles), ARTICLES_PER_DAY)
                for article in articles[:count]:
                    # Extract only the desired fields.
                    headline = article.headline if article.headline else ""
                    summary  = article.summary if article.summary else ""
                    created_at = article.created_at.isoformat() if article.created_at else ""
                    updated_at = article.updated_at.isoformat() if article.updated_at else ""
                    
                    # Combine headline and summary for sentiment analysis.
                    combined_text = f"{headline} {summary}"
                    sentiment, confidence = predict_sentiment(combined_text)
                    
                    # Format the confidence scores to three decimal places.
                    conf_negative = "{:.3f}".format(confidence[0])
                    conf_neutral  = "{:.3f}".format(confidence[1])
                    conf_positive = "{:.3f}".format(confidence[2])
                    
                    # Prepare the row dictionary.
                    row = {
                        "Headline": headline,
                        "Summary": summary,
                        "CreatedAt": created_at,
                        "UpdatedAt": updated_at,
                        "Sentiment": sentiment,
                        "Confidence_Negative": conf_negative,
                        "Confidence_Neutral": conf_neutral,
                        "Confidence_Positive": conf_positive,
                    }
                    writer.writerow(row)
                    # Flush immediately to save the data.
                    csvfile.flush()
                    print(f"Article added for {current_day.date()}: {headline} (Sentiment: {sentiment})")
            else:
                print(f"No article found for {current_day.date()}.")
        except Exception as e:
            print(f"Error retrieving news for {current_day.date()}: {e}")

print(f"\nCSV file '{CSV_FILENAME}' created successfully.")
