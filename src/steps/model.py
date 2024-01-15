import pandas as pd
from src.components.modelbuilding import Modeltraining
import logging
from zenml import step
import mlflow
from zenml.client import Client

experiment_tracker = Client().active_stack.experiment_tracker




@step(experiment_tracker=experiment_tracker.name)
def model_train(data:pd.DataFrame,epochs:int,lstm_nodes: int,lstm_nodes2: int, dropout_rate: float,vector_feature:int,folder_path:str,data_path:str):

    try:
         
         mlflow.autolog()

         model=Modeltraining()
         result=model.train(data,epochs,lstm_nodes,lstm_nodes2,dropout_rate,vector_feature,folder_path,data_path)
         logging.info("model trained successfully")

         return result


    except Exception as e:
        logging.error(e)
        raise e

#run the pipeline check zenml for registration has done or not



