import yfinance as yf
import csv
from datetime import datetime
from transformers import pipeline

# 1) Set up the sentiment analysis pipeline
pipe = pipeline("text-classification", model="mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")

# 2) Create a Ticker object for the desired stock symbol (e.g., "AAPL")
ticker = yf.Ticker("TSLA")

# 3) Retrieve up to 50 news articles (Yahoo may provide fewer if not available)
news = ticker.get_news(count=100)

# 4) Define output file path
csv_file_path = 'news_articles.csv'

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
            sentiment_result = pipe(combined_text)[0]["label"]

            # Write the row to CSV (title, summary, pub_date, link, sentiment)
            writer.writerow([title, summary, pub_date, link, sentiment_result])

    print(f"CSV file '{csv_file_path}' created successfully with up to {len(news)} articles.")