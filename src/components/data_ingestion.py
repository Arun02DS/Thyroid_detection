from src.logger import logging
from src.exception import ThyDetectException
from src import utils
from src.entity import config_entity
from src.entity import artifact_entity
import pandas as pd 
import numpy as np 
import os,sys
from sklearn.model_selection import train_test_split
from src.config import col_names

"""
This script prepare for data ingestion phase of training pipeline.

"""

class DataIngestion:

    def __init__(self,data_ingestion_config:config_entity.DataIngestionConfig):
        try:
            logging.info(f"{'..>==>..'*4} Data Ingestion {'..<==<..'*4}")
            self.data_ingestion_config=data_ingestion_config
        except Exception as e:
            raise ThyDetectException(e, sys)

    def initiate_data_ingestion(self)->artifact_entity.DataIngestionArtifact:
        try:
            df:pd.DataFrame = utils.get_collection_as_dataframe(
                database_name=self.data_ingestion_config.database_name,
                collection_name=self.data_ingestion_config.collection_name)

            logging.info("preparing dataset for further phases of pipeline.")
            #Save data in feature store
            df.replace(to_replace='?',value=np.NAN,inplace=True)
            #making changes to datset as per notebook
            df=utils.make_df(df=df,col_names=col_names)

            #create a feature store folder
            feature_store_dir = os.path.dirname(self.data_ingestion_config.feature_store_file_path)
            os.makedirs(feature_store_dir,exist_ok=True)

            #saving dataframe into csv format
            logging.info("saving dataset in feature store.")
            df.to_csv(path_or_buf=self.data_ingestion_config.feature_store_file_path, header=True, index=False)
            logging.info(f"rows and columns in df: {df.shape}")


            #Splitting data into train and test
            logging.info("splitting dataset into train and test dataset.")
            train_df,test_df = train_test_split(df,test_size=self.data_ingestion_config.test_size,random_state=42)
            logging.info(f"rows and columns in train df: {train_df.shape} \n rows and columns in test df: {test_df.shape}")


            #creating dataset directory,if already do not exist
            dataset_dir = os.path.dirname(self.data_ingestion_config.train_file_path)
            os.makedirs(dataset_dir,exist_ok=True)

            # Saving df into train and test in dataset folder
            logging.info("saving train and test into dataset folder.")
            train_df.to_csv(path_or_buf=self.data_ingestion_config.train_file_path, header=True, index=False)
            test_df.to_csv(path_or_buf=self.data_ingestion_config.test_file_path, header=True, index=False)

            logging.info("preparing artifact.")
            #defining artifact
            data_ingestion_artifact=artifact_entity.DataIngestionArtifact(
                feature_store_file_path=self.data_ingestion_config.feature_store_file_path,
                train_file_path=self.data_ingestion_config.train_file_path,
                test_file_path=self.data_ingestion_config.test_file_path)
            
            logging.info(f"data ingestion artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact

        except Exception as e:
            raise ThyDetectException(e, sys)
    



