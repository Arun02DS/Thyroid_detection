from src.logger import logging
from src.exception import ThyDetectException
from typing import Optional
import os,sys
from src.entity.config_entity import TRANSFORMER_OBJECT_FILE_NAME,LABEL_ENCODER_OBJECT_FILE_NAME,MODEL_FILE_NAME


class ModelResolver:

    def __init__(self,model_registry:str="saved_models",
    transformer_dir_name="transformed_obj",
    label_encoder_dir_name="encoder_obj",
    model_dir_name="Previous_model"):
        
        self.model_registry=model_registry
        os.makedirs(model_registry,exist_ok=True)
        self.transformer_dir_name=transformer_dir_name
        self.label_encoder_dir_name=label_encoder_dir_name
        self.model_dir_name=model_dir_name
    
    def get_latest_dir_path(self)->Optional[str]:
        """
        ============================================================================================
        Description: This function provides the latest path in model registry.

        Return: latest directory path
        =============================================================================================
        """
        try:
            dir_names=os.listdir(self.model_registry)
            if len(dir_names) ==0:
                return None
            dir_names = list(map(int,dir_names))
            latest_dir_name = max(dir_names)
            return os.path.join(self.model_registry,f"{latest_dir_name}")
        except Exception as e:
            raise ThyDetectException(e, sys)

    def get_latest_transformer_path(self):
        """
        ============================================================================================
        Description: This function provides the latest transformer path in model registry.

        Return: latest transformer path in directory.
        =============================================================================================
        """
        try:
            latest_dir = self.get_latest_dir_path()
            if latest_dir is None:
                raise Exception("Transformer is not available.")
            return os.path.join(latest_dir,self.transformer_dir_name,TRANSFORMER_OBJECT_FILE_NAME)

        except Exception as e:
            raise ThyDetectException(e, sys)

    def get_latest_encoder_path(self):
        """
        ============================================================================================
        Description: This function provides the latest label encoder path in model registry.

        Return: latest label encoder path in directory.
        =============================================================================================
        """
        try:
            latest_dir= self.get_latest_dir_path()
            if latest_dir is None:
                raise Exception("Latest encoder is not available")
            return os.path.join(latest_dir,self.label_encoder_dir_name,LABEL_ENCODER_OBJECT_FILE_NAME)
            
        except Exception as e:
            raise ThyDetectException(e, sys)

    def get_latest_model_path(self):
        """
        ============================================================================================
        Description: This function provides the latest model path in model registry.

        Return: latest model path in directory.
        =============================================================================================
        """
        try:
            latest_dir = self.get_latest_dir_path()
            if latest_dir is None:
                raise Exception("Model is not available.")
            return os.path.join(latest_dir,self.model_dir_name,MODEL_FILE_NAME)

        except Exception as e:
            raise ThyDetectException(e, sys)

    def get_latest_save_dir_path(self):
        """
        ============================================================================================
        Description: This function provides the latest path to save in directory.

        Return: latest save path in directory.
        =============================================================================================
        """
        try:
            latest_dir = self.get_latest_dir_path()
            if latest_dir is None:
                return os.path.join(self.model_registry,f"{0}")
            latest_dir_num = int(os.path.basename(self.get_latest_dir_path()))
            return os.path.join(self.model_registry,f"{latest_dir_num+1}")
            
        except Exception as e:
            raise ThyDetectException(e, sys)

    def get_latest_save_transformer_path(self):
        """
        ============================================================================================
        Description: This function provides the latest path to save in directory for transformer.

        Return: latest save path in directory for transformer.
        =============================================================================================
        """
        try:
            latest_dir = self.get_latest_save_dir_path()
            return os.path.join(latest_dir,self.transformer_dir_name,TRANSFORMER_OBJECT_FILE_NAME)
        except Exception as e:
            raise ThyDetectException(e, sys)
    
    def get_latest_save_model_path(self):
        """
        ============================================================================================
        Description: This function provides the latest path to save in directory for model.

        Return: latest save path in directory for model.
        =============================================================================================
        """
        try:
            latest_dir = self.get_latest_save_dir_path()
            return os.path.join(latest_dir,self.model_dir_name,MODEL_FILE_NAME)
        except Exception as e:
            ThyDetectException(e, sys)
    
    def get_latest_save_encoder_path(self):
        """
        ============================================================================================
        Description: This function provides the latest path to save in directory for label encoder.

        Return: latest save path in directory for label encoder.
        =============================================================================================
        """
        try:
            latest_dir = self.get_latest_save_dir_path()
            return os.path.join(latest_dir,self.label_encoder_dir_name,LABEL_ENCODER_OBJECT_FILE_NAME)
        except Exception as e:
            raise ThyDetectException(e, sys)
    


