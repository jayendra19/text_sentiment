from src.components.textpreprocessing import clean_text
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
from tensorflow.keras.models import load_model
import pickle


# List of model paths
model_paths = [
    r"C:\Users\jayen\Text-sentiment-analysis-general-purpose\newmodels\emotions_model.h5",
    r"C:\Users\jayen\Text-sentiment-analysis-general-purpose\newmodels\Forsentiment3_model.h5",
    r"C:\Users\jayen\Text-sentiment-analysis-general-purpose\newmodels\Mental_health_model.h5",
    r"C:\Users\jayen\Text-sentiment-analysis-general-purpose\newmodels\sarcasm_dataset_updated_model.h5"
]

tokenizer_paths=[
    r"C:\Users\jayen\Text-sentiment-analysis-general-purpose\newmodels\emotions_tokenizer.pickle",
    r"C:\Users\jayen\Text-sentiment-analysis-general-purpose\newmodels\Forsentiment3_tokenizer.pickle",
    r"C:\Users\jayen\Text-sentiment-analysis-general-purpose\newmodels\Mental_health_tokenizer.pickle",
    r"C:\Users\jayen\Text-sentiment-analysis-general-purpose\newmodels\sarcasm_dataset_updated_tokenizer.pickle"


]

# Load all models
models = [load_model(model_path) for model_path in model_paths]

tokenizers = [pickle.load(open(tokenizer_path, 'rb')) for tokenizer_path in tokenizer_paths]

# Define label mappings
label_mappings = [
    {0: 'Sadness', 1: 'Joy', 2: 'Lovely', 3: 'Anger', 4: 'Fear',5:'Suprised'},  # emotion_mapping
    {0: 'Neutral', 1: 'Positive', 2: 'Negative'},  # sentiment_mapping
    {0: 'anxiety', 1: 'Anxious', 2: 'Depression',3:"Good", 4:"Lonely", 5: "Normal", 6: "Stressed", 7: "Suicidel"},  # mental_health_mapping
    {0: 'Not Sarcastic', 1: 'Sarcastic'}  # sarcasm_mapping
]
#tokenizers=Tokenizer
def classify_text(text):
    # Get predictions from each model
    results = {}
    for model, tokenizer, model_name, label_mapping in zip(models, tokenizers, ['Emotion','Sentiment', 'Mental Health', 'Sarcasm'], label_mappings):
        # Preprocess the text
        texts = clean_text(text)
        input_sequence = tokenizer.texts_to_sequences([texts])
        padded_input = pad_sequences(input_sequence, maxlen=200)

        prediction = model.predict(padded_input)
        # Get the class with the highest probability
        predicted_class = np.argmax(prediction)
        # Map the predicted class to its label
        predicted_label = label_mapping[predicted_class]
        results[model_name] = predicted_label
        
    print(results)

    return results