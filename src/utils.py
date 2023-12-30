import pandas as pd
from src.config import mongo_client
from src.logger import logging
from src.exception import ThyDetectException
import sys,os
import yaml
from typing import List

"""
This script contains the most common functions used regularly in this project.

"""
def get_collection_as_dataframe(database_name:str,collection_name:str)->pd.DataFrame:

    """
    =====================================================================

    Description: This function returns collection as dataframe.

    Input:  database_name:database name
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
    

def make_df(df: pd.DataFrame, col_names: List[str]) -> pd.DataFrame:
    """
    ========================================================================
    Convert all values for thyroid detected to "Yes" in a Pandas DataFrame.

    Input:
    - df: Pandas DataFrame
    - col_names: List of column names to manupulate/ drop na values in rows

    Return:
    Pandas DataFrame with updated thyroid detection values

    ========================================================================
    """
    try:
        # Replacing all values in specified columns with 'Yes' if they are not 'No' or NaN
        df[col_names[0]] = df[col_names[0]].apply(lambda x: 'Yes' if pd.notnull(x) and x != 'No' else x)

        # Dropping rows where gender information is missing
        logging.info("dropping rows where gender is not known/provided")
        df.dropna(subset=[col_names[1]], inplace=True)
        df.reset_index(drop=True, inplace=True)
        
        return df

    except Exception as e:
        raise ThyDetectException(e, sys)

def convert_column_dtype(df:pd.DataFrame)->pd.DataFrame:
    """
    ===================================================================================

    Description: This function convert all dtype to numerical if they are int or float
    else it will be object type.

    Input : Pandas dataframe
    Return: Pandas dataframe

    =======================================================================================

    """
    try:
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='ignore')
        return df
    except Exception as e:
        raise ThyDetectException(e, sys)

def write_yaml_file(file_path,data:dict):
    """
    ===========================================================================================
    Description: This fuction writes file in yaml format in a specified file path.
    ===========================================================================================
    
    """
    try:
        file_dir = os.path.dirname(file_path)
        os.makedirs(file_dir,exist_ok=True)

        with open(file_path,"w") as file_writer:
            yaml.dump(data,file_writer)
            
    except Exception as e:
        raise ThyDetectException(e, sys)