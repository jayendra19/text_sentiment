import pandas as pd
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from abc import ABC,abstractmethod
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Bidirectional
from sklearn.model_selection import train_test_split
import numpy as np
import os
from keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
import logging
import pickle

class Model(ABC):

    @abstractmethod
    def train(self,data:pd.DataFrame, epochs:int,lstm_nodes: int,stm_nodes2: int, dropout_rate: float,vector_feature:int,folder_path:str,data_path:str):
        ''' take a Pandas DataFrame as input, perform some operations on the data within the method, and then return either a DataFrame or a Series.'''
        pass 



class Modeltraining(Model):


    def train(self,data:pd.DataFrame,epochs:int,lstm_nodes: int,lstm_nodes2: int, dropout_rate: float,vector_feature:int,folder_path:str,data_path:str):
        
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(data['preprocessed'])
        sequences = tokenizer.texts_to_sequences(data['preprocessed'])

        max_length = 200  # Define your maximum sequence length
        padded_sequences = pad_sequences(sequences, maxlen=max_length)#THIS WILL BE MY ACTUAL DATA THAT'LL TRAIN TEST AND SPLIT 
        word_index = tokenizer.word_index
        num_words = len(word_index)+1

        ## Creating model
        embedding_vector_features=vector_feature ##features representation each word wil get convert into a vector of 300
        model=Sequential()
        model.add(Embedding(num_words,embedding_vector_features,input_length=max_length))
        model.add(Dropout(dropout_rate))
        model.add(Bidirectional(LSTM(lstm_nodes,return_sequences=True)))
        model.add(Dropout(dropout_rate))
        model.add(Bidirectional(LSTM(lstm_nodes2)))

        num_classes = len(set(data['label']))

        # For binary classification
        if num_classes == 2: 
            activation = 'sigmoid'
            loss = 'binary_crossentropy'
            model.add(Dense(1,activation=activation))
            model.compile(loss=loss,optimizer='adam',metrics=['accuracy'])

        # For multiclass classification
        else:
            activation = 'softmax'
            loss = 'sparse_categorical_crossentropy'

            model.add(Dense(num_classes,activation=activation))
            model.compile(loss=loss,optimizer='adam',metrics=['accuracy'])
            logging.info("Model is created")

        X_final=np.array(padded_sequences)
        y_final = np.array(data['label'])
        X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.2, random_state=42)
        print(X_train.shape)
        print(y_train.shape)
        print(y_train)


        early_stopping = EarlyStopping(monitor='val_loss', patience=4)
    
        model.fit(X_train,y_train,epochs=epochs,batch_size=32, validation_data=(X_test,y_test),callbacks=[early_stopping])
        logging.info("Modele trained")
        

        if num_classes ==2:
            logging.info("Evaluation started")
            y_pred_prob=model.predict(X_test)
            # Applying different thresholds 
            thresholds = np.arange(0, 1, 0.01)
            #finding best thresholds for accurary 
            scores = []
            for t in thresholds:
                y_pred = (y_pred_prob > t).astype(int)
                scores.append(accuracy_score(y_test, y_pred))

            # Geting the threshold with the highest score
            best_threshold = thresholds[np.argmax(scores)]
            y_pred = np.where(y_pred_prob > best_threshold, 1, 0)  # Apply a threshold

            print("Accuracy Score:\n", accuracy_score(y_test,y_pred))
            logging.info("Modele trained")
            

        else:
            logging.info("Evaluaion started")
            y_pred = model.predict(X_test)
            predicted_labels = [max(enumerate(pred), key=lambda x: x[1])[0] for pred in y_pred]
            print("Confusion Matrix:\n", confusion_matrix(y_test,predicted_labels))
            print("\nClassification Report:\n", classification_report(y_test,predicted_labels))
            logging.info("Modele trained")
            
        folder_path = folder_path

        # Create the folder if it doesn't exist
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        # Extract dataset name from the file path
        dataset_name = os.path.splitext(os.path.basename(data_path))[0]

        # Save your trained model with a unique name
        model_path = os.path.join(folder_path, f'{dataset_name}_model.h5')
        model.save(model_path)

        # Save the tokenizer
        tokenizer_path = os.path.join(folder_path, f'{dataset_name}_tokenizer.pickle')
        with open(tokenizer_path, 'wb') as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return model.summary() #history which is dictionary containing the loss values and metrics so it history.history used to acceses dict from the history object. 

        








