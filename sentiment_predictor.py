

# sentiment_predictor.py

import pickle
import re
import spacy

# Load spaCy Portuguese model

nlp = spacy.load("pt_core_news_sm", disable=["parser","ner"])

# Preprocessing function

def preprocess(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)      # Remove URLs
    text = re.sub(r"<.*?>", " ", text)               # Remove HTML tags
    text = re.sub(r"[^a-zà-ú\s]", " ", text)         # Keep only letters and Portuguese characters
    text = re.sub(r"\s+", " ", text).strip()         # Remove extra spaces

    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    return " ".join(tokens)


# Load saved model and TF-IDF vectorizer

with open("sentiment_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("tfidf_vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)

# Take user input and predict sentiment

if __name__ == "__main__":
    review_input = "Péssima qualidade, não recomendo"
    clean_review = [preprocess(review_input)]
    X_input = tfidf.transform(clean_review)
    prediction = model.predict(X_input)
    print(f"Predicted Sentiment: {prediction[0]}")

