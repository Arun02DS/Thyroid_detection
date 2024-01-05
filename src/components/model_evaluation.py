from src.entity import config_entity,artifact_entity
from src.Prediction import ModelResolver
from src.logger import logging
from src.exception import ThyDetectException
import os,sys
from src.utils import load_object,convert_column_dtype,inverse_trans,trans
import pandas as pd
from src.config import col_names
from sklearn.metrics import f1_score


"""
This script prepare for data evaluation phase of training pipeline.

"""

class ModelEvaluation:

    def __init__(self,model_evaluation_config:config_entity.ModelEvaluationConfig,
    data_ingestion_artifact:artifact_entity.DataIngestionArtifact,
    data_transformation_artifact:artifact_entity.DataTransformationArtifact,
    model_trainer_artifact:artifact_entity.ModelTrainerArtifact):

        try:
            logging.info(f"{'..>==>..'*4} Model Evaluation {'..<==<..'*4}")
            self.model_evaluation_config=model_evaluation_config
            self.data_ingestion_artifact=data_ingestion_artifact
            self.data_transformation_artifact=data_transformation_artifact
            self.model_trainer_artifact=model_trainer_artifact
            self.model_resolver = ModelResolver()
        except Exception as e:
            raise ThyDetectException(e, sys)

    
    def initiate_model_evaluation(self)->artifact_entity.ModelEvaluationArtifact:
        try:
            #check if model is present in saved model
            logging.info(f"Checking if model is present in saved model folder")
            latest_dir_path=self.model_resolver.get_latest_dir_path()
            if latest_dir_path is None:
                model_evaluation_artifact = artifact_entity.ModelEvaluationArtifact(is_model_accepted=True, 
                                            improved_accuracy=None)
                logging.info(f"model evaluation artifact: {model_evaluation_artifact}")
                return model_evaluation_artifact
            
            logging.info("Model is present in saved model folder")
            logging.info("locations for transformer,encoder and model")
            # Transforemer,encoder and model path
            transformer_path = self.model_resolver.get_latest_transformer_path()
            model_path =self.model_resolver.get_latest_model_path()
            label_encoder_path = self.model_resolver.get_latest_encoder_path()

            #previously Trained objects
            logging.info('loading previously Trained objects')
            transformer = load_object(file_path=transformer_path)
            model = load_object(file_path=model_path)
            label_encoder = load_object(file_path=label_encoder_path)

            #current objects
            logging.info("loading current objects")
            current_transformer = load_object(self.data_transformation_artifact.transformation_object_path)
            current_label_encoder = load_object(self.data_transformation_artifact.label_encoder_path)
            current_model = load_object(self.model_trainer_artifact.model_path)

            # Loading test df
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            test_df = convert_column_dtype(df=test_df)
            object_cols = test_df.select_dtypes(include='object').columns.tolist()
            
            test_df = trans(cols=object_cols, encoder=label_encoder, df=test_df)
            y_true = test_df[col_names[0]]

            #Checking accuracy using previous model

            logging.info("checking accuracy using previous model")
            input_feature_name = list(transformer.feature_names_in_)
            input_arr = transformer.transform(test_df[input_feature_name])
            y_pred = model.predict(input_arr)

            y_pred_inverse = inverse_trans(col=col_names[0], encoder=label_encoder, y_pred=y_pred[:5])
            logging.info(f"Prediction using previous model: {y_pred_inverse},{y_true[:5].values.tolist()}")
            Previous_model_score = f1_score(y_true=y_true,y_pred=y_pred)
            logging.info(f"Accuracy using previous model: {Previous_model_score}")


            test_df_current = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            test_df_current = convert_column_dtype(df=test_df)
            object_cols = test_df_current.select_dtypes(include='object').columns.tolist()
            #checking with current model
            
            test_df_current = trans(cols=object_cols, encoder=current_label_encoder, df=test_df_current)
            y_true_current = test_df_current[col_names[0]]

            logging.info("checking accuracy using current model")
            input_feature_name_current = list(current_transformer.feature_names_in_)
            input_arr_current = current_transformer.transform(test_df_current[input_feature_name_current])
            y_pred_current = current_model.predict(input_arr_current)

            y_pred_inverse_current = inverse_trans(col=col_names[0], encoder=current_label_encoder, y_pred=y_pred_current[:5])
            logging.info(f"Prediction using current model: {y_pred_inverse_current},{y_true_current[:5].values.tolist()}")
            current_model_score = f1_score(y_true=y_true_current,y_pred=y_pred_current)
            logging.info(f"Accuracy using current model: {current_model_score}")

            if current_model_score <= Previous_model_score:
                logging.info("Current trained model is not better than previously trained model")
                model_evaluation_artifact = artifact_entity.ModelEvaluationArtifact(is_model_accepted=False
                , improved_accuracy = current_model_score-Previous_model_score)

                logging.info(f"model evaluation artifact: {model_evaluation_artifact}")
                return model_evaluation_artifact
            else:
                logging.info(f"current trained model is better than previous trained model")
                model_evaluation_artifact = artifact_entity.ModelEvaluationArtifact(is_model_accepted=True
                , improved_accuracy = current_model_score-Previous_model_score)

                logging.info(f"model evaluation artifact: {model_evaluation_artifact}")
                return model_evaluation_artifact

        except Exception as e:
            raise ThyDetectException(e, sys)

