import sys,os
import pandas
from src.logger import logging
from src.exception import ThyDetectException
from src.utils import get_collection_as_dataframe



if __name__=="__main__":
     try:
          get_collection_as_dataframe(database_name='thyroid', collection_name='thydetect')
     except Exception as e:
          print(e)

