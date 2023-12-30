
from src.entity import config_entity,artifact_entity
from src.logger import logging
from src.exception import ThyDetectException
import sys,os
import pandas as pd
from typing import Optional
from scipy.stats import ks_2samp
import numpy as np 
from src import utils
from src.config import col_names

"""
This script will run after data ingestion phase.

"""

class DataValidation:

    def __init__(self, data_validation_config:config_entity.DataValidationConfig,
                data_ingestion_artifact:artifact_entity.DataIngestionArtifact):

        try:
            logging.info(f"{'..>==>..'*4} Data Validation {'..<==<..'*4}")
            self.data_validation_config = data_validation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.validation_report=dict()
        except Exception as e:
            raise ThyDetectException(e, sys)
    


    def drop_missing_values_column(self,df:pd.DataFrame,report_key_name:str)->Optional[pd.DataFrame]:
        """
        ====================================================================================================

        Description: This function takes dataframe as input and drop the columns which are not meeting the
        threshold criterion.

        Return: It will return pandas dataframe or None as all columns could be dropped.

        =====================================================================================================

        """
        try:
            #Threshold criterion
            threshold = self.data_validation_config.na_threshold
            # null percentage 
            null_report = df.isna().sum()/df.shape[0]

            #Dropping column names
            logging.info(f"Selecting columns which do not met {threshold} threshold criterion.")
            drop_column_names = null_report[null_report>threshold].index

            #dropped all columns
            logging.info(f"columns dropping: {drop_column_names}")
            self.validation_report[report_key_name] = list(drop_column_names)
            df.drop(list(drop_column_names),axis=1,inplace=True)

            logging.info(f"All columns are dropped which do not met criterion, if not None will be returned")
            if len(df.columns)==0:
                return None
            return df
        except Exception as e:
            raise ThyDetectException(e, sys)

    def is_required_column_exist(self,base_df:pd.DataFrame,current_df:pd.DataFrame,report_key_name:str)->bool:
        try:
            
            #base and current columns
            base_columns = base_df.columns
            current_columns = current_df.columns

            missing_columns=[]
            for base_column in base_columns:
                if base_column not in current_columns:
                    logging.info(f"columns: {base_column} is not in current column")
                    missing_column.append(base_column)
            logging.info(f"missing columns _if_exist: {missing_columns}")
            if len(missing_columns) > 0:
                logging.info(f"missing column report w.r.t base and current column")
                self.validation_report[report_key_name] = list(missing_columns)
                return False
            return True

        except Exception as e:
            raise ThyDetectException(e, sys)

    def data_drift(self,base_df:pd.DataFrame,current_df:pd.DataFrame,report_key_name:str):
        try:
            drift_report=dict()
            base_columns=base_df.columns
            current_columns=current_df.columns

            for base_column in base_columns:
                #Creating base data and current data from base df column info
                base_data,current_data = base_df[base_column],current_df[base_column]

                logging.info(f"Hypothesis {base_column}: {base_data.dtype},{current_data.dtype}")
                same_distribution = ks_2samp(base_data,current_data)
            
                if same_distribution.pvalue > 0.05:
                    # we are accepting null hypothesis
                    drift_report[base_column]={
                        'pvalues':float(same_distribution.pvalue),
                        'same_distribution':True,
                        'Null_Hypothesis': "Accepted"
                    }
                else:
                    drift_report[base_column]={
                        'pvalues':float(same_distribution.pvalue),
                        'same_distribution':False,
                        'Null_Hypothesis': "Rejected"
                    }
            self.validation_report[report_key_name]=drift_report

        except Exception as e:
            raise ThyDetectException(e, sys)
    

    def initiate_data_validation(self)->artifact_entity.DataValidationArtifact:
        try:
            logging.info(f"Fetching base df")
            base_df = pd.read_csv(self.data_validation_config.base_file_path)
            #replacing "?" into np.NAN
            logging.info(f"replacing '?' into np.NAN")
            base_df.replace(to_replace="?",value=np.NAN,inplace=True)
            #dropping columns
            logging.info(f"Dropping columns from base dataset")
            base_df=utils.make_df(df=base_df,col_names=col_names)
            base_df = self.drop_missing_values_column(df=base_df, report_key_name="Missing_columns_in_base_dataset")
            

            #Fetching train and test dataset
            logging.info(f"Fetching train df")
            train_df = pd.read_csv(self.data_ingestion_artifact.train_file_path)
            logging.info(f"Fetching test df")
            test_df  = pd.read_csv(self.data_ingestion_artifact.test_file_path)

            #dropping columns from test and train dataset
            logging.info(f"Dropping columns from train dataset")
            train_df = self.drop_missing_values_column(df=train_df, report_key_name="Missing_columns_in_train_dataset")
            logging.info(f"Dropping columns from test dataset")
            test_df = self.drop_missing_values_column(df=test_df, report_key_name="Missing_columns_in_test_dataset")

            #converting all columns dtype
            logging.info(f"Checking and converting all dtype")
            base_df = utils.convert_column_dtype(df=base_df)
            train_df = utils.convert_column_dtype(df=train_df)
            test_df = utils.convert_column_dtype(df=test_df)

            #compairing base df and current df
            logging.info(f"If all columns are present in train df")
            train_df_column_status = self.is_required_column_exist(base_df=base_df, current_df=train_df, report_key_name="status_on_train__how_many_columns_exist")
            logging.info(f"If all columns are present in test df")
            test_df_column_status = self.is_required_column_exist(base_df=base_df, current_df=test_df, report_key_name="status_on_test__how_many_columns_exist")

            #checking for data drift
            if train_df_column_status:
                logging.info(f"As columns are present in train df hence detecting data drift in train df")
                self.data_drift(base_df=base_df, current_df=train_df, report_key_name="data drift on train dataset")

            if test_df_column_status:
                logging.info(f"As columns are present in test df hence detecting data drift in test df")
                self.data_drift(base_df=base_df, current_df=test_df, report_key_name="data drift on test dataset")

            #creating report 
            logging.info("write a report on yaml file")
            utils.write_yaml_file(file_path=self.data_validation_config.report_file_path,
                                    data=self.validation_report)
            

            data_validation_artifact = artifact_entity.DataValidationArtifact(report_file_path=self.data_validation_config.report_file_path)
            logging.info(f"data validation Artifact: {data_validation_artifact}")
            return data_validation_artifact

        except Exception as e:
            raise ThyDetectException(e, sys)

