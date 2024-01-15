import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from abc import ABC,abstractclassmethod
from typing import Union
import logging
import neattext as nt
import contractions
from sklearn.preprocessing import LabelEncoder
# Download NLTK resources (do it once)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))



class Preprocessing(ABC):

    @abstractclassmethod
    def handle_data(self:pd.DataFrame)->Union[pd.DataFrame,pd.Series]:
        pass



def clean_text(text):
        # Initialize NeatText TextFrame and NLTK WordNetLemmatizer
        # Handle contractions
        # Check for empty strings
        '''if not text or len(text) < 2: 
            return text'''
        text = contractions.fix(text)
        docx = nt.TextFrame(text)
        

        # Clean text using NeatText
        cleaned_text = docx.remove_puncts()
        cleaned_text = cleaned_text.remove_special_characters()
        cleaned_text = cleaned_text.remove_userhandles()
        cleaned_text = cleaned_text.remove_urls()
        cleaned_text = cleaned_text.remove_dates()
        cleaned_text = cleaned_text.remove_emails()
        cleaned_text = cleaned_text.remove_emojis()

        # Convert text to lowercase
        cleaned_text = str(cleaned_text).lower()

        # Remove stop words
        stop_words = set(stopwords.words('english'))
        cleaned_text = " ".join(word for word in cleaned_text.split() if word not in stop_words)

        # Lemmatize words
        cleaned_text = " ".join(lemmatizer.lemmatize(word) for word in cleaned_text.split())

        return cleaned_text



class Textpreprocessing(Preprocessing):

    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        '''data: pd.DataFrame: It expects a parameter named data of type pd.DataFrame. This indicates that the method is designed to handle Pandas DataFrame objects.'''
        try:
            logging.info('received data')
            print(data)
            print(data.info())

            # Check for null values
            if data.isnull().values.any():
                data = data.dropna()

            # Checking if labels are numerical or categorical
            if not pd.api.types.is_numeric_dtype(data['label']):
                # Apply label encoding to categorical labels
                label_encoder = LabelEncoder()
                data['label'] = label_encoder.fit_transform(data['label'])

            # Apply the clean_text function
            data['preprocessed'] = data['text'].apply(clean_text)

            logging.info("preprocessed has been done")

            return data

        except Exception as e:
            logging.error(e)
            raise e

















