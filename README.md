# Udemy NLP Bootcamp Projects

End-to-end NLP projects completed as part of the **Complete Data Science, Machine Learning, Deep Learning & NLP Bootcamp** on Udemy.

**Instructor:** Krish Naik & KRISHAI Technologies Private Limited  
**Duration:** 99 hours  
**Completed:** April 3, 2026  
**Certificate:** [UC-58b30962-da2f-41e5-bcaa-58c57422ba65](https://ude.my/UC-58b30962-da2f-41e5-bcaa-58c57422ba65)

---

## Repository Structure

```
├── Udemy_Task1.ipynb                      # Project 1: NewsAI — News Classification & Summarization
├── Udemy_Task2.ipynb                      # Project 2: MovieLens — Sentiment Analysis & Theme Extraction
├── Vivan_Kushal_NLP_Project_Summary.pdf   # Full Project Portfolio Summary
├── Udemy_Certificate.pdf                  # Certificate of Completion
└── README.md
```

---

## Project 1 – NewsAI: News Classification & Summarization

**Role:** NLP Engineer at a News Aggregation Startup  
**Dataset:** AG News Corpus — 127,000 articles (120k train / 7.6k test)

### Objective
Build a smart news pipeline to automatically categorise incoming articles into topics (Politics, Sports, Tech, Health) and generate concise extractive summaries for quick article previews — demo-ready for an investor pitch.

### Pipeline (8 Steps)

| Step | Description |
|---|---|
| 1 | Environment setup — scikit-learn, SpaCy, datasets, matplotlib |
| 2 | Data loading — AG News CSV, label mapping, title + description combined, Health category added via keyword detection |
| 3 | Text preprocessing — lowercasing, URL removal, punctuation stripping |
| 4 | Model training — TF-IDF (50k features, bigrams) + Logistic Regression (C=5.0) |
| 5 | Extractive summarization — SpaCy sentence scoring, top-N sentences, ~65–70% compression |
| 6 | End-to-end pipeline — single `news_pipeline()` call returns category, confidence, probabilities, and summary |
| 7 | Visualisation — F1 bar chart, category distribution, confusion matrix heatmap |
| 8 | Investor scorecard — pitch-ready summary printout |

### Results

| Metric | Value |
|---|---|
| Model Accuracy | ~92% |
| Training Data | 120,000 news articles |
| Categories | Politics, Sports, Tech, Health |
| Summary Compression | ~68% |

### How to Run

1. Open `Udemy_Task1.ipynb` in Google Colab or Jupyter.
2. Run all cells sequentially — no API key required.

```bash
pip install scikit-learn spacy matplotlib seaborn
python -m spacy download en_core_web_sm
```

---

## Project 2 – MovieLens: Sentiment Analysis & Theme Extraction

**Role:** Data Scientist at a Movie Review Platform  
**Dataset:** IMDB Movie Reviews (HuggingFace) — 50,000 reviews (25k train / 25k test)

### Objective
Automatically detect sentiment (Positive / Negative / Neutral) in user movie reviews and extract recurring themes to improve recommendations, identify complaints early, and drive marketing decisions.

### Three-Part Approach

**Task 1 — Sentiment Classification**

| Method | Approach | Accuracy |
|---|---|---|
| VADER (Rule-Based) | Compound score thresholds, no training needed | ~82% |
| TF-IDF + Logistic Regression | 30k features, bigrams, trained on 25k reviews | ~92% |

**Task 2 — Keyphrase Extraction & Topic Modelling**

- **RAKE** — multi-word phrase extraction using word co-occurrence statistics
- **YAKE** — statistical keyword uniqueness scoring, top 5 keyphrases per review
- **LDA Topic Modelling** — applied to 5,000 reviews, discovered 6 hidden topics: Filmmaking, Horror/Thriller, Romance/Drama, Action, Comedy, Performances & Awards

**Task 3 — Visualisation Dashboard**

4-panel Matplotlib dashboard:
- Sentiment distribution pie chart
- Positive vs Negative coefficient bar chart (top 8 words each)
- WordCloud of negative review terms
- VADER score histogram by sentiment
- 22 complaint/praise signals tracked per review

### Results

| Metric | Value |
|---|---|
| VADER Accuracy | ~82% (no training) |
| ML Accuracy | ~92% (trained) |
| LDA Topics Discovered | 6 |
| Signals Tracked | 22 (11 praise + 11 complaint) |

### How to Run

1. Open `Udemy_Task2.ipynb` in Google Colab or Jupyter.
2. Run all cells sequentially — dataset loads automatically via HuggingFace.

```bash
pip install scikit-learn nltk vaderSentiment rake-nltk yake datasets matplotlib seaborn wordcloud
```

---

## Technologies Used

| Category | Libraries |
|---|---|
| Core NLP | SpaCy, NLTK, VADER, RAKE, YAKE |
| ML / Modelling | scikit-learn, TF-IDF, Logistic Regression, LDA |
| Data | HuggingFace Datasets, AG News CSV |
| Visualisation | Matplotlib, Seaborn, WordCloud |
| Environment | Python, Google Colab |

---

## Skills Demonstrated

- End-to-end NLP pipeline development (data → preprocessing → modelling → evaluation → visualisation)
- Both rule-based (VADER) and learned (TF-IDF + LR) approaches to text classification
- Unsupervised topic modelling with LDA on unlabelled data
- Production-minded single-function pipeline wrappers (`news_pipeline()`, `analyse_review()`)
- Business-framed notebooks with stakeholder-ready summaries and markdown explanations
- Real-world benchmark datasets at scale (127k + 50k samples)
