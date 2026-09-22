# CodeAlpha Data Analytics Internship — Sentiment Analysis

## Overview

This project was completed as part of the **CodeAlpha Data Analytics Internship**.

The objective was to perform sentiment analysis on book descriptions using two different natural language processing approaches: **VADER** and **TextBlob**.

## Dataset

The analysis was performed on book descriptions from the scraped book dataset.

The project contains sentiment results for **1,000 book records**.

## Sentiment Analysis Methods

### VADER

VADER sentiment classification produced:

| Sentiment      | Count |
| -------------- | ----: |
| Positive       |   680 |
| Negative       |   303 |
| Neutral        |    15 |
| No description |     2 |

### TextBlob

TextBlob sentiment classification produced:

| Sentiment      | Count |
| -------------- | ----: |
| Positive       |   732 |
| Negative       |    96 |
| Neutral        |   170 |
| No description |     2 |

The agreement between VADER and TextBlob sentiment classifications was **66.0%**.

The project also calculated average VADER sentiment scores across different book-rating groups.

## Technologies Used

* Python
* Pandas
* NLTK
* VADER Sentiment
* TextBlob
* Matplotlib
* Seaborn

## Files

```text id="7f2k1m"
codealpha_sentimentanalysis/
├── README.md
├── sentiment.py
├── books_data.csv
├── books_sentiment.csv
└── charts/
    ├── sentiment_distribution.png
    └── sentiment_vs_rating.png
```

## Setup

Install the required libraries:

```bash id="3x8v4p"
pip install pandas nltk textblob matplotlib seaborn
```

The script automatically downloads the required VADER lexicon when needed.

## Usage

Run the sentiment analysis script with:

```bash id="6n1r8q"
python sentiment.py
```

The script generates:

* `books_sentiment.csv`
* `charts/sentiment_distribution.png`
* `charts/sentiment_vs_rating.png`

## Output

The resulting dataset contains sentiment scores and classifications generated using both VADER and TextBlob.

The visualizations show the distribution of sentiment classifications and the relationship between sentiment scores and book ratings.

## Internship Task

**CodeAlpha Data Analytics Internship — Task 4: Sentiment Analysis**
