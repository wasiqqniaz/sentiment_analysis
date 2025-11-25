# 🇧🇷 Olist Customer Review Sentiment Analysis

This project develops a **Natural Language Processing (NLP)** model to perform sentiment analysis on customer reviews from the Brazilian e-commerce platform, Olist. The goal is to classify review messages into **positive**, **neutral**, or **negative** sentiments using a Logistic Regression classifier trained on **TF-IDF** features.

## Project Workflow & NLP Pipeline

The project follows a standard machine learning pipeline, with a strong focus on the **NLP workflow** for Portuguese text data.

### 1\. Data Loading & Cleaning

  * **Source:** `olist_order_reviews_dataset.csv`
  * **Initial Cleaning:** Dropping irrelevant columns (`review_id`, `order_id`, etc.) and focusing on `review_score` and `review_comment_message`.

### 2\. Handling Missing Reviews (Imputation)

  * **Strategy:** Missing or empty review messages are imputed with a default Portuguese sentiment phrase (e.g., "muito bom" for score 5, "muito ruim" for score 1) based on their `review_score`. This ensures all records contribute to the model training.

### 3\. **The Core NLP Workflow: Text Preprocessing**

This is the most critical NLP step, using the specialized **spaCy** Portuguese model (`pt_core_news_sm`).

  * **Steps:**
      * Convert text to **lowercase**.
      * Remove **URLs** and **HTML tags**.
      * Filter to keep only **letters, accents, and spaces** (removing numbers and most punctuation).
      * **Tokenization** (breaking text into words).
      * **Lemmatization** (reducing words to their dictionary root form, e.g., "correndo" $\rightarrow$ "correr").
      * **Stopword Removal** (removing common, low-information words like "a", "o", "de", "que").

### 4\. Sentiment Label Creation

  * **Target Mapping:** The `review_score` (1-5) is mapped to three sentiment classes:
      * **Negative:** Scores 1 & 2
      * **Neutral:** Score 3
      * **Positive:** Scores 4 & 5

### 5\. **NLP Feature Extraction (TF-IDF)**

  * **Technique:** **Term Frequency-Inverse Document Frequency (TF-IDF)** is used to convert the clean, preprocessed text into a numerical feature matrix (`X_tfidf`).
  * **Configuration:** `max_features=5000` and `ngram_range=(1, 2)` (considering unigrams and bigrams) to capture word and phrase importance.

### 6\. Model Training & Evaluation

  * **Model:** **Logistic Regression** (a strong baseline for text classification).
  * **Split:** 80% Train, 20% Test, using `stratify=Y` to maintain class balance.
  * **Performance:** The model achieved high overall **accuracy (0.93)**, with strong performance in the **positive (F1: 0.96)** and acceptable performance in the **negative (F1: 0.83)** classes. The lower recall for the **neutral** class is common in sentiment analysis due to ambiguous language.

| Sentiment | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **Negative** | 0.82 | 0.84 | 0.83 | 2915 |
| **Neutral** | 0.92 | 0.57 | 0.71 | 1636 |
| **Positive** | 0.95 | 0.98 | 0.96 | 15294 |
| **Accuracy** | - | - | 0.93 | 19845 |

### 7\. Deployment

  * The trained `model` and `tfidf_vectorizer` are saved using `pickle`.
  * A deployment script (`sentiment_predictor.py`) is provided for making real-time predictions.

## Prerequisites

To run this project, you will need:

  * Python 3.x
  * The following libraries:
      * `pandas`
      * `numpy`
      * `scikit-learn`
      * `spacy`
      * `re`
      * `pickle`

### Installation

1.  Install the required Python packages:
    ```bash
    pip install pandas numpy scikit-learn spacy
    ```
2.  Install the specific **spaCy Portuguese language model**:
    ```bash
    python -m spacy download pt_core_news_sm
    ```

## How to Use

### Predicting New Reviews

The `sentiment_predictor.py` script demonstrates how to load the saved model artifacts and make a prediction.

1.  Ensure you have run the notebook to generate `sentiment_model.pkl` and `tfidf_vectorizer.pkl`.
2.  Run the prediction script from your terminal:
    ```bash
    python sentiment_predictor.py
    ```
3.  The output will be the predicted sentiment for the hardcoded example:
    ```
    Predicted Sentiment: negative
    ```
    *(The example review in the script is "Péssima qualidade, não recomendo," which is expected to be negative.)*
