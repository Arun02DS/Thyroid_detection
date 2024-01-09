import os,sys
from datetime import datetime
from src.logger import logging
from src.exception import ThyDetectException


FILE_NAME='thyroid.csv'
TRAIN_FILE_NAME='train.csv'
TEST_FILE_NAME='test.csv'
TRANSFORMER_OBJECT_FILE_NAME='transformer.pkl'
LABEL_ENCODER_OBJECT_FILE_NAME='label_encoder.pkl'
MODEL_FILE_NAME = 'model.pkl'

class TrainingPipelineConfig:

    def __init__(self):
        self.artifact_dir= os.path.join(os.getcwd(),"artifact",f"{datetime.now().strftime('%d%m%Y__%H%M%S')}")
        os.makedirs(self.artifact_dir,exist_ok=True)

class DataIngestionConfig:

    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        try:
            self.database_name = 'thyroid'
            self.collection_name = 'thydetect'
            self.data_ingestion_dir = os.path.join(training_pipeline_config.artifact_dir,'data_ingestion')
            self.feature_store_file_path = os.path.join(self.data_ingestion_dir,'feature_store',FILE_NAME)
            self.train_file_path = os.path.join(self.data_ingestion_dir,'dataset',TRAIN_FILE_NAME)
            self.test_file_path = os.path.join(self.data_ingestion_dir,'dataset',TEST_FILE_NAME)
            self.test_size=0.20
        except Exception as e:
            raise ThyDetectException(e, sys)
    
    def to_dict(self)->dict:
        try:
            return self.__dict__
        except Exception as e:
            raise ThyDetectException(e, sys)


class DataValidationConfig:

    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        self.data_validation_dir = os.path.join(training_pipeline_config.artifact_dir,'data_validation')
        self.report_file_path = os.path.join(self.data_validation_dir,'report.yaml')
        self.na_threshold=0.20
        self.base_file_path = os.path.join('thyroid_data_.csv')

class DataTransformationConfig:
    
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        self.data_transformation_dir = os.path.join(training_pipeline_config.artifact_dir,'data_transformation')
        self.transformation_object_path = os.path.join(self.data_transformation_dir,'transformer',TRANSFORMER_OBJECT_FILE_NAME)
        self.transformation_train_path = os.path.join(self.data_transformation_dir,'transformed',TRAIN_FILE_NAME.replace("csv","npz"))
        self.transformation_test_path = os.path.join(self.data_transformation_dir,'transformed',TEST_FILE_NAME.replace("csv","npz"))
        self.label_encoder_path = os.path.join(self.data_transformation_dir,'label_encoder',LABEL_ENCODER_OBJECT_FILE_NAME)

class ModelTrainerConfig:

    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        self.model_trainer_dir= os.path.join(training_pipeline_config.artifact_dir,'model_trainer')
        self.model_path = os.path.join(self.model_trainer_dir,"model",MODEL_FILE_NAME)
        self.expected_score = 0.80
        self.overfitting_threshold = 0.10

class ModelEvaluationConfig:
    
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        self.diff_threshold = 0.01

class ModelPusherConfig:

    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        self.model_pusher_dir = os.path.join(training_pipeline_config.artifact_dir,'model_pusher')
        self.saved_model_dir = os.path.join("saved_models")
        self.pusher_model_dir = os.path.join(self.model_pusher_dir,"objects_saved")
        self.pusher_model_path = os.path.join(self.pusher_model_dir,MODEL_FILE_NAME)
        self.pusher_transformer_path = os.path.join(self.pusher_model_dir,TRANSFORMER_OBJECT_FILE_NAME)
        self.pusher_label_encoder_path = os.path.join(self.pusher_model_dir,LABEL_ENCODER_OBJECT_FILE_NAME)
