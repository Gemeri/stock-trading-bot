import yfinance as yf
import csv
from datetime import datetime
from transformers import pipeline

from transformers import AutoTokenizer, AutoModelForSequenceClassification

# setup model and tokenizer
model_name = "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# 2) Create a Ticker object for the desired stock symbol (e.g., "AAPL")
ticker = yf.Ticker("TSLA")

# 3) Retrieve up to 50 news articles (Yahoo may provide fewer if not available)
news = ticker.get_news(count=20)

# 4) Define output file path
csv_file_path = 'news_articles.csv'


# Function to perform sentiment analysis
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model(**inputs)
    sentiment_class = outputs.logits.argmax(dim=1).item()
    sentiment_mapping = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    predicted_sentiment = sentiment_mapping.get(sentiment_class, 'Unknown')
    return predicted_sentiment, outputs.logits.softmax(dim=1)[0].tolist()

if not news:
    print("No news articles available.")
else:
    with open(csv_file_path, mode='w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write a header row
        writer.writerow(["Title", "Summary", "Published", "Link", "Sentiment"])

        for article in news:
            content  = article.get('content', {})
            title    = content.get('title', 'No title')
            summary  = content.get('summary', 'No summary')
            pub_date = content.get('pubDate', 'No publish time')
            link     = content.get('canonicalUrl', {}).get('url', 'No link')

            # Convert pub_date to a human-readable format if possible
            if pub_date != 'No publish time':
                try:
                    pub_date = datetime.strptime(pub_date, '%Y-%m-%dT%H:%M:%SZ')
                    pub_date = pub_date.strftime('%Y-%m-%d %H:%M:%S')
                except ValueError:
                    pass  # Keep the original string if parsing fails

            # Combine Title + Summary for sentiment analysis
            combined_text = f"{title} {summary}"

            sentiment, confidence_scores = predict_sentiment(combined_text)

            print(f"prediction: {sentiment}")
            print(f"confidence_scores: {confidence_scores}")

            # Format the confidence scores to 3 decimal places (and transform to string)
            confidence_scores[0] = "{:.3f}".format(confidence_scores[0])
            confidence_scores[1] = "{:.3f}".format(confidence_scores[1])
            confidence_scores[2] = "{:.3f}".format(confidence_scores[2])


            # Write the row to CSV (title, summary, pub_date, link, negative, neutral, positive
            writer.writerow([title, summary, pub_date, link, confidence_scores[0], confidence_scores[1], confidence_scores[2]])

    print(f"CSV file '{csv_file_path}' created successfully with up to {len(news)} articles.")