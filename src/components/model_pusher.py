from src.logger import logging
from src.exception import ThyDetectException
from src.entity import config_entity,artifact_entity
import os,sys
from src.Prediction import ModelResolver
from src.utils import load_object,save_object


class ModelPusher:

    def __init__(self,model_pusher_config:config_entity.ModelPusherConfig,
    data_transformation_artifact:artifact_entity.DataTransformationArtifact,
    model_trainer_artifact:artifact_entity.ModelTrainerArtifact):
        try:
            logging.info(f"{'..>==>..'*4} Model Pusher {'..<==<..'*4}")
            self.model_pusher_config=model_pusher_config
            self.data_transformation_artifact=data_transformation_artifact
            self.model_trainer_artifact=model_trainer_artifact
            self.model_resolver = ModelResolver(model_registry=self.model_pusher_config.saved_model_dir)
        
        except Exception as e:
            raise ThyDetectException(e,sys)

    def initiate_model_pusher(self)->artifact_entity.ModelPusherArtifact:
        try:
            # Loading tranformer,encoder amd model object
            logging.info("Loading tranformer,encoder amd model object")
            transformer = load_object(file_path=self.data_transformation_artifact.transformation_object_path)
            label_encoder = load_object(file_path=self.data_transformation_artifact.label_encoder_path)
            model = load_object(file_path=self.model_trainer_artifact.model_path)

            #saving obects in model Pusher directory
            logging.info("Saving objects in model pusher directory")
            save_object(file_path=self.model_pusher_config.pusher_transformer_path, obj=transformer)
            save_object(file_path=self.model_pusher_config.pusher_model_path, obj=model)
            save_object(file_path=self.model_pusher_config.pusher_label_encoder_path, obj=label_encoder)

            #Saving models in save model directory
            logging.info("Saving models in save model directory")
            transformer_path = self.model_resolver.get_latest_save_transformer_path()
            model_path = self.model_resolver.get_latest_save_model_path()
            label_encoder_path = self.model_resolver.get_latest_save_encoder_path()

            save_object(file_path=transformer_path, obj=transformer)
            save_object(file_path=model_path, obj=model)
            save_object(file_path=label_encoder_path, obj=label_encoder)

            model_pusher_artifact = artifact_entity.ModelPusherArtifact(
                pusher_model_dir=self.model_pusher_config.pusher_model_dir, 
                saved_model_dir=self.model_pusher_config.saved_model_dir)
            
            logging.info(f"Model Pusher Artifact: {model_pusher_artifact}")
            return model_pusher_artifact


        except Exception as e:
            raise ThyDetectException(e, sys)