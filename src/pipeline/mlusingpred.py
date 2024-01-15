import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline
from src.components.textpreprocessing import clean_text

# Assuming you have trained scikit-learn models for sentiment, mental health, emotion, and sarcasm
# Replace these with your actual models
model_emo = joblib.load(r"C:\Users\jayen\Text-sentiment-analysis-general-purpose\models\logistic_regression_model_emotions.pkl")
model_mental = joblib.load(r"C:\Users\jayen\Text-sentiment-analysis-general-purpose\models\logistic_regression_model_mentalhealth.pkl")
model_sar = joblib.load(r"C:\Users\jayen\Text-sentiment-analysis-general-purpose\models\logistic_regression_model.pkl")
model_senti = joblib.load(r"C:\Users\jayen\Text-sentiment-analysis-general-purpose\models\logistic_regression_model_sentiment.pkl")

# Load the vectorizer during inference
vectorizer = joblib.load(r'C:\Users\jayen\Text-sentiment-analysis-general-purpose\models\vectorizer.joblib')

# Assuming you have defined these mapping dictionaries
sentiment_mapping = {0: 'Neutral', 1: 'Positive', 2: 'Negative'}
mental_health_mapping = {0: 'anxiety', 1: 'Anxious', 2: 'Depression',3:"Good", 4:"Lonely", 5: "Normal", 6: "Stressed", 7: "Suicidel"}
emotion_mapping = {0: 'Sadness', 1: 'Joy', 2: 'Lovely', 3: 'Anger', 4: 'Fear',5:'Suprised'}
sarcasm_mapping = {0: 'Not Sarcastic', 1: 'Sarcastic'}

def classify_text(text):
    # Preprocess the text
    preprocessed_text = clean_text(text)

    # Vectorize the text
    text_vectorized = vectorizer.transform([preprocessed_text])

    # Get predictions from each model
    sentiment = model_emo.predict(text_vectorized)
    mental_health = model_mental.predict(text_vectorized)
    emotion = model_sar.predict(text_vectorized)
    sarcasm = model_senti.predict(text_vectorized)

    # Convert numerical indices to original labels
    sentiment_labels = sentiment_mapping.get(sentiment[0], 'Unknown')
    mental_health_labels = mental_health_mapping.get(mental_health[0], 'Unknown')
    emotion_labels = emotion_mapping.get(emotion[0], 'Unknown')
    sarcasm_labels = sarcasm_mapping.get(sarcasm[0], 'Unknown')

    # Collect and return the results
    results = {
        'Sentiment': sentiment_labels,
        'Mental Health': mental_health_labels,
        'Emotion': emotion_labels,
        'Sarcasm': sarcasm_labels
    }
    return results





