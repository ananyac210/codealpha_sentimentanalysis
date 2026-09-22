"""
CodeAlpha Data Analytics Internship - Task 4: Sentiment Analysis
--------------------------------------------------------------------
Classifies each book's description (scraped by scraper.py) as
Positive, Negative, or Neutral using two complementary NLP approaches:

  1. VADER (nltk) - lexicon + rule-based, tuned for short informal text
  2. TextBlob      - lexicon-based polarity/subjectivity scoring

Both are run so you can compare them (a common EDA-style step in
sentiment analysis write-ups) and discuss agreement/disagreement.

Requirements:
    pip install pandas nltk textblob matplotlib seaborn
    python -m textblob.download_corpora
    (the script also auto-downloads the VADER lexicon on first run)

Usage:
    python sentiment.py   (run AFTER scraper.py has created books_data.csv
                            with a 'description' column)

Output:
    books_sentiment.csv         (original data + sentiment scores/labels)
    charts/sentiment_distribution.png
    charts/sentiment_vs_rating.png
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

CSV_PATH = "books_data.csv"
OUT_CSV = "books_sentiment.csv"
OUT_DIR = "charts"

sns.set_theme(style="whitegrid")


def ensure_vader():
    """Download VADER lexicon if not already present."""
    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        nltk.download("vader_lexicon")


def classify_vader(text, sia):
    if not isinstance(text, str) or not text.strip():
        return None, "No description"
    score = sia.polarity_scores(text)["compound"]
    if score >= 0.05:
        label = "Positive"
    elif score <= -0.05:
        label = "Negative"
    else:
        label = "Neutral"
    return score, label


def classify_textblob(text):
    if not isinstance(text, str) or not text.strip():
        return None, "No description"
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.05:
        label = "Positive"
    elif polarity < -0.05:
        label = "Negative"
    else:
        label = "Neutral"
    return polarity, label


def plot_sentiment_distribution(df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    order = ["Positive", "Neutral", "Negative"]
    sns.countplot(x="vader_label", data=df, order=order, hue="vader_label",
                   palette="viridis", legend=False, ax=axes[0])
    axes[0].set_title("Sentiment Distribution (VADER)")
    axes[0].set_xlabel("Sentiment")
    axes[0].set_ylabel("Number of Books")

    sns.countplot(x="textblob_label", data=df, order=order, hue="textblob_label",
                   palette="magma", legend=False, ax=axes[1])
    axes[1].set_title("Sentiment Distribution (TextBlob)")
    axes[1].set_xlabel("Sentiment")
    axes[1].set_ylabel("Number of Books")

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/sentiment_distribution.png", dpi=150)
    plt.close()


def plot_sentiment_vs_rating(df):
    plt.figure(figsize=(8, 5))
    sns.boxplot(x="rating", y="vader_score", data=df, hue="rating",
                palette="crest", legend=False)
    plt.title("VADER Sentiment Score by Star Rating")
    plt.xlabel("Star Rating")
    plt.ylabel("VADER Compound Sentiment Score")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/sentiment_vs_rating.png", dpi=150)
    plt.close()


def main():
    ensure_vader()
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_csv(CSV_PATH)
    if "description" not in df.columns:
        raise ValueError(
            "No 'description' column found. Re-run the updated scraper.py "
            "to include book descriptions before running sentiment analysis."
        )

    sia = SentimentIntensityAnalyzer()

    vader_results = df["description"].apply(lambda t: classify_vader(t, sia))
    df["vader_score"] = vader_results.apply(lambda x: x[0])
    df["vader_label"] = vader_results.apply(lambda x: x[1])

    tb_results = df["description"].apply(classify_textblob)
    df["textblob_score"] = tb_results.apply(lambda x: x[0])
    df["textblob_label"] = tb_results.apply(lambda x: x[1])

    df.to_csv(OUT_CSV, index=False)

    print("=" * 70)
    print("SENTIMENT ANALYSIS SUMMARY")
    print("=" * 70)
    print("\nVADER label counts:")
    print(df["vader_label"].value_counts())
    print("\nTextBlob label counts:")
    print(df["textblob_label"].value_counts())

    agree = (df["vader_label"] == df["textblob_label"]).mean() * 100
    print(f"\nAgreement between VADER and TextBlob: {agree:.1f}%")

    print("\nAverage VADER sentiment score by star rating:")
    print(df.groupby("rating")["vader_score"].mean().round(3))

    plot_sentiment_distribution(df)
    plot_sentiment_vs_rating(df)

    print(f"\nSaved {OUT_CSV}")
    print(f"Saved charts/sentiment_distribution.png")
    print(f"Saved charts/sentiment_vs_rating.png")


if __name__ == "__main__":
    main()
