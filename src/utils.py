import pandas as pd
from src.config import mongo_client
from src.logger import logging
from src.exception import ThyDetectException
import sys


"""
This script contains the most common functions used regularly in this project.

"""
def get_collection_as_dataframe(database_name:str,collection_name:str)->pd.DataFrame:

    """
    =====================================================================

    Description: This function returns collection as dataframe.
    database_name:database name
    collection_name: collection name

    Return: Pandas dataframe of collections.
    ======================================================================

    """
    try:

        logging.info(f"Reading data from mongo db , database{database_name} and collection name {collection_name}")
        df = pd.DataFrame(list(mongo_client[database_name][collection_name].find()))
        logging.info(f"column names: {df.columns}")
        if "_id" in df.columns:
            logging.info(f"_id column found and dropping from df")
            df = df.drop("_id",axis=1)
            logging.info("_id column dropped")
        logging.info(f"Number of rows and column in df: {df.shape}")
        return df

    
    except Exception as e:
        raise ThyDetectException(e, sys)
    

