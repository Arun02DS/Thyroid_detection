import sys,os
import pandas
from src.logger import logging
from src.exception import ThyDetectException
from src.utils import get_collection_as_dataframe
from src.entity.config_entity import DataIngestionConfig
from src.entity import config_entity
from src.components import data_ingestion



if __name__=="__main__":
     try:
          training_pipeline_config=config_entity.TrainingPipelineConfig()
          data_ingestion_config = DataIngestionConfig(training_pipeline_config=training_pipeline_config)
          print(data_ingestion_config.to_dict())

          data_ingestion = data_ingestion.DataIngestion(data_ingestion_config=data_ingestion_config)
          print(data_ingestion.initiate_data_ingestion())
     except Exception as e:
          print(e)

