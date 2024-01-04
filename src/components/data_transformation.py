from src.entity import config_entity,artifact_entity
from src.logger import logging
from src.exception import ThyDetectException
import os,sys
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from imblearn.combine import SMOTETomek
import pandas as pd 
import numpy as np 
from src import utils
from src.config import col_names
from sklearn.preprocessing import LabelEncoder


class DataTransformation:

    def __init__(self,data_transformation_config:config_entity.DataTransformationConfig,
                data_ingestion_artifact:artifact_entity.DataIngestionArtifact):
        try:
            logging.info(f"{'..>==>..'*4} Data Transformation {'..<==<..'*4}")
            self.data_transformation_config = data_transformation_config
            self.data_ingestion_artifact = data_ingestion_artifact

        except Exception as e:
            raise ThyDetectException(e,sys)
    
    @classmethod

    def get_data_transformation_object(cls)->Pipeline:
        """
        ====================================================================================
        Description: This class method provides a pipeline obeject which imputes and scale
        the data.

        Return: Pipeline

        ====================================================================================
        
        """

        try:
            simple_imputer = SimpleImputer(strategy='median')
            robust_scaler = RobustScaler()

            pipeline = Pipeline(steps=[
                ('imputer',simple_imputer),
                ('scaler',robust_scaler)
            ])

            return pipeline

        except Exception as e:
            raise ThyDetectException(e, sys)
    
    def initiate_data_transformation(self,)->artifact_entity.DataTransformationArtifact:
        try:
            train_df = pd.read_csv(self.data_ingestion_artifact.train_file_path)
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            

            #changing dtpye for the dataset
            train_df=utils.convert_column_dtype(df=train_df)
            test_df=utils.convert_column_dtype(df=test_df)
            

            # Separate numeric and object columns
            numeric_cols = train_df.select_dtypes(include=np.number).columns.tolist()
            object_cols = train_df.select_dtypes(include='object').columns.tolist()

            logging.info(f"We have {len(numeric_cols)} numerical features: {numeric_cols}")
            logging.info(f"We have {len(object_cols)} categorical features: {object_cols}")

            #One hot Encoding the train and test data
            label_encoder = {}
            for col in object_cols:
                label_encode = LabelEncoder()
                combined_data = pd.concat([train_df[col], test_df[col]], axis=0)
                label_encode.fit(combined_data)
                train_df[col] = label_encode.fit_transform(train_df[col])
                test_df[col] = label_encode.transform(test_df[col])
                label_encoder[col]=label_encode

            input_feature_train_df = train_df.drop(col_names[0],axis=1)
            input_feature_test_df = test_df.drop(col_names[0],axis=1)

            target_feature_train_df = train_df[col_names[0]]
            target_feature_test_df = test_df[col_names[0]]

            #Transforming input features
            transformation_pipeline=DataTransformation.get_data_transformation_object()
            transformation_pipeline.fit(input_feature_train_df)
            input_feature_train_arr = transformation_pipeline.transform(input_feature_train_df)
            input_feature_test_arr = transformation_pipeline.transform(input_feature_test_df)

            #Balancing the dataset
            smt = SMOTETomek(sampling_strategy='minority',random_state=42)
            logging.info(f"Before resampling training set: Input - {input_feature_train_arr.shape}, Target-{target_feature_train_df.shape}")
            input_feature_train_arr,target_feature_train_arr = smt.fit_resample(input_feature_train_arr,target_feature_train_df)
            logging.info(f"Before resampling training set: Input - {input_feature_train_arr.shape}, Target-{target_feature_train_arr.shape}")

            logging.info(f"Before resampling testing set: Input - {input_feature_test_arr.shape}, Target-{target_feature_test_df.shape}")
            input_feature_test_arr,target_feature_test_arr = smt.fit_resample(input_feature_test_arr,target_feature_test_df)
            logging.info(f"Before resampling testing set: Input - {input_feature_test_arr.shape}, Target-{target_feature_test_arr.shape}")

            #combining dataset
            train_arr = np.c_[input_feature_train_arr,target_feature_train_arr]
            test_arr = np.c_[input_feature_test_arr,target_feature_test_arr]

            #saving the object and arrays
            logging.info(f"Saving the transformation object and array")
            utils.save_numpy_arr_data(file_path=self.data_transformation_config.transformation_train_path,
                                         array=train_arr)
            
            utils.save_numpy_arr_data(file_path=self.data_transformation_config.transformation_test_path
                                        , array=test_arr)

            utils.save_object(file_path=self.data_transformation_config.transformation_object_path, 
                                        obj=transformation_pipeline)
            
            utils.save_object(file_path=self.data_transformation_config.label_encoder_path,
                                          obj=label_encoder)
    
            data_transformation_artifact = artifact_entity.DataTransformationArtifact(
                transformation_object_path=self.data_transformation_config.transformation_object_path, 
                transformation_train_path=self.data_transformation_config.transformation_train_path, 
                transformation_test_path=self.data_transformation_config.transformation_test_path,
                label_encoder_path=self.data_transformation_config.label_encoder_path)

            logging.info(f"Transformation Artifact: {data_transformation_artifact}")
            return data_transformation_artifact

        except Exception as e:
            raise ThyDetectException(e, sys)
