import mlflow
from zenml import pipeline
from src.steps.ingest import ingest_df
from src.steps.preprocessing import text_preprocessing
from src.steps.model import model_train



# List of paths to your datasets
data_paths = [
       r'C:\Users\jayen\text_sentiment\artifacts\emotions.csv',
       r'C:\Users\jayen\text_sentiment\artifacts\Forsentiment3.csv',
       r'C:\Users\jayen\text_sentiment\artifacts\Mental_health.csv',
       r'C:\Users\jayen\text_sentiment\artifacts\sarcasm_dataset_updated.csv'
    ]
@pipeline(enable_cache=False)
def training_pipeline():
            # Iterate over each dataset
            for data_path in data_paths:
            
                with mlflow.start_run():
                        try:

                            # Pass raw_data_path to ingest_data
                            df = ingest_df(data_path=data_path,limit_rows=50000)
                            print("Data ingested successfully.")

                            dataframe= text_preprocessing(df)
                            print("Data preprocessed successfully.")

                            # Create a ModelNameConfig instance and pass it to train_model
                            #An epoch is one complete pass through the entire training dataset.
                            result= model_train(dataframe,5,50,25,0.3,100,r'C:\Users\jayen\text_sentiment\newmodels',data_path)
                            print("Data pipeline done successfully.")

                        finally:
                              mlflow.end_run()

