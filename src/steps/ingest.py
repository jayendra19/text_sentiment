import logging
import pandas as pd
from zenml import step
import logging
from zenml.client import Client
from typing import Optional


experiment_tracker = Client().active_stack.experiment_tracker



class IngestData:
    """
    Data ingestion class which ingests data from the source and returns a DataFrame.
    """

    def __init__(self, data_path: str,limit_rows: Optional[int] = None):
        """Initialize the data ingestion class."""
        self.data_path = data_path
        self.limit_rows=limit_rows

    def get_data(self) -> pd.DataFrame:
         return pd.read_csv(self.data_path,nrows=self.limit_rows)
        
    


@step()
def ingest_df(data_path: str,limit_rows: Optional[int] = None) -> pd.DataFrame:
    """
    Args:
        data_path: str
    Returns:
        df: pd.DataFrame
    """
    try:
        ingest_data = IngestData(data_path,limit_rows=limit_rows)

        df = ingest_data.get_data()
        return df
    except Exception as e:
        logging.error(e)
        raise e


#df1 = ingest_df("dataset1.csv", limit_rows=1000000)
#df2 = ingest_df("dataset2.csv", limit_rows=None)  