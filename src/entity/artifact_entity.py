from dataclasses import dataclass


@dataclass
class DataIngestionArtifact:
    feature_store_file_path:str
    train_file_path:str
    test_file_path:str


@dataclass
class DataValidationArtifact:
    report_file_path:str

@dataclass
class DataTransformationArtifact:
    transformation_object_path:str
    transformation_train_path:str
    transformation_test_path:str
    label_encoder_path:str

class ModelTrainerArtifact:...
class ModelEvaluationArtifact:...
class ModelPusherArtifact:...