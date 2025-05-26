# Context Extraction Modeling for Financial News Sentiment Analysis

## 🧾 Overview

This project explores the use of named entity recognition (NER), large language models (LLMs), and unsupervised topic modeling to build a context-aware sentiment analysis pipeline for financial news.

Unlike traditional sentiment classifiers that rate entire headlines or articles, this project first extracts **relevant entities** and then uses a **prompted LLM (GPT-3.5)** to generate **contextual summaries** grounded in those entities. These summaries are clustered, topic modelled then fed into **FinBERT**, a financial sentiment classifier, yielding more reliable and interpretable sentiment labels.

The goal is to simulate how a financial analyst might interpret an article: focusing on the **who**, **what**, and **why**—and connecting this to **market sentiment** and **topic structure** over time.

The result is a highly explainable and analytically powerful pipeline capable of:
- Extracting key market contexts from long-form financial articles
- Running sentiment analysis at a more granular level
- Visualizing sentiment volatility over time
- Clustering and labeling financial themes across thousands of articles

## 🚧 Problem Statement

Most financial sentiment analysis tools rely on classifying entire articles or headlines without understanding **why** a given sentiment arises. This often leads to:
- Misclassifications when the article contains multiple events or conflicting tones
- Poor explainability and lack of actionable insights
- Inability to aggregate sentiment by topic or financial entity

This project solves that by:
- Using NER to extract the **relevant entities**
- Applying an LLM prompt to generate **contextual summaries**
- Classifying sentiment **based on that context**, not the full article

This enables better downstream tasks such as:
- **Tracking sentiment volatility** over time
- **Understanding market reactions** to specific companies, events, or policy changes
- **Clustering financial discourse** into labeled, sentiment-aware topics

## 🧠 Pipeline Architecture

The modeling pipeline includes the following stages:

1. **Data Collection**:
   - Used 53,000 articles from a NASDAQ financial news dataset (including titles, dates, and article bodies)

2. **NER & Entity Cleaning**:
   - Extracted entities using `dslim/bert-base-NER` via HuggingFace pipeline
   - Cleaned low-confidence or fragmented entities

3. **Contextual Summary Generation**:
   - Prompted GPT-3.5 to summarize articles **around detected entities**
   - Enforced token budget using tiktoken to truncate long texts

4. **Sentiment Classification**:
   - Used `FinBERT` to classify sentiment of the **generated context summaries**
   - Output includes Positive, Negative, or Neutral labels

5. **Embedding & Clustering**:
   - Encoded summaries using `MiniLM-L6-v2`
   - Reduced dimensionality with UMAP
   - Clustered using HDBSCAN

6. **Topic Modeling & Labeling**:
   - Used BERTopic to discover and name 87 unique financial topics
   - Updated topic names with OpenAI-based topic summarizer

7. **Visualization & Trend Analysis**:
   - Temporal sentiment trends (e.g., pre/post 2020)
   - Topic frequency and sentiment volatility charts

## 🧪 Results

- **87** distinct financial topics discovered using BERTopic
- **Contextual sentiment** labels:  
  - Positive: 34,290  
  - Negative: 10,538  
  - Neutral: 8,081
- Improved signal quality using **context-based sentiment classification** instead of full-article tone
- Identified major shifts and **volatility in sentiment trends** over time (2008–2023)
- Topic timelines revealed **which financial events drove public opinion** in specific periods

## 🎯 Key Features

- 🧠 **NER + LLM Prompting**: Combines structured entity detection with natural language summarization
- 💬 **Contextual Sentiment Modeling**: Classifies tone on focused summaries, not full articles
- 🧩 **Unsupervised Topic Clustering**: Reveals macro trends and sentiment drivers without labels
- 📈 **Sentiment Volatility Tracker**: Measures the emotional turbulence in financial media over time
- 📊 **Topic Evolution Visualization**: See how themes like "Earnings", "Cannabis Stocks", or "Monetary Policy" change over the years
