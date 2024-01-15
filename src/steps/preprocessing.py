import logging
import pandas as pd
from src.components.textpreprocessing import Textpreprocessing
from zenml import step





@step
def text_preprocessing(data:pd.DataFrame)->pd.DataFrame:

    try:
        text=Textpreprocessing()
        processed_df=text.handle_data(data)
        logging.info("text_preprocessing has been done")

        return processed_df
    
    except Exception as e:
        logging.error(e)
        raise e




    





