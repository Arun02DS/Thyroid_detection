from src.entity import config_entity,artifact_entity
from src.logger import logging
from src.exception import ThyDetectException
import os,sys
from xgboost import XGBClassifier
from src import utils
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from src.config import pram_grid,col_names


"""
This script prepare for data trainer phase of training pipeline.

"""

class ModelTrainer:

    def __init__(self,model_trainer_config:config_entity.ModelTrainerConfig,
    data_transformation_artifact:artifact_entity.DataTransformationArtifact):
        try:
            logging.info(f"{'..>==>..'*4} Model Trainer {'..<==<..'*4}")
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise ThyDetectException(e, sys)

    def train_model(self,x,y,best_params=None):
        """
        =================================================================================================

        Description: This function fit X and y to a model.

        Return :  Model

        =================================================================================================
    
        """
        try:
            if best_params is None:
                xgb_clf = XGBClassifier()
            else:
                xgb_clf = XGBClassifier(**best_params)
            xgb_clf.fit(x,y)
            return xgb_clf
        except Exception as e:
            raise ThyDetectException(e, sys)

    def parameter_tuning(self,X,y,estimator,param_grid):
        """
        =================================================================================================

        Description: This function fit X and y to a model with GridsearchCV to fine tune the model.

        Return :  best params , best score

        =================================================================================================
    
        """
        gsearch = GridSearchCV(estimator, param_grid,scoring='f1',n_jobs=4, cv=5)
        gsearch.fit(X,y)
        return gsearch.best_params_,gsearch.best_score_
    
    def initiate_model_trainer(self)->artifact_entity.ModelTrainerArtifact:
        try:
            #Loading train and test array
            logging.info(f" Train and test array loaded")
            train_arr = utils.load_numpy_array_data(file_path=self.data_transformation_artifact.transformation_train_path)
            test_arr = utils.load_numpy_array_data(file_path=self.data_transformation_artifact.transformation_test_path)

            #splitting train and test array into X and y
            logging.info(f"splitting the train and test arrays")
            X_train,y_train = train_arr[:,:-1],train_arr[:,-1]
            X_test,y_test = test_arr[:,:-1],test_arr[:,-1]

            # Creating model
            logging.info("creating model")
            model = self.train_model(x=X_train,y=y_train,best_params=None)

            yhat_train = model.predict(X_train)
            f1_train_score = f1_score(y_true=y_train,y_pred=yhat_train)
            logging.info(f" f1 train :{f1_train_score}")

            yhat_test = model.predict(X_test)
            f1_test_score = f1_score(y_true=y_test,y_pred=yhat_test)
            logging.info(f"f1 test: {f1_test_score}")

            #Tuning the model
            best_params,best_score = self.parameter_tuning(X=X_train, y=y_train, estimator=model, param_grid=pram_grid)
            logging.info(f"best params: {best_params}, best score : {best_score}")
            
            logging.info(f"Creating model after Grid search")
            model_gs = self.train_model(x=X_train,y=y_train,best_params=best_params)

            yhat_train_gs = model_gs.predict(X_train)
            f1_train_score_gs = f1_score(y_true=y_train,y_pred=yhat_train_gs)
            logging.info(f" f1 train gs :{f1_train_score_gs}")

            yhat_test_gs = model_gs.predict(X_test)
            f1_test_score_gs = f1_score(y_true=y_test,y_pred=yhat_test_gs)
            logging.info(f" f1 test gs:{f1_test_score_gs}")
            

            if f1_test_score_gs > f1_test_score:
                logging.info(f" model_gs is best model with given parameters")
                logging.info(f" model train score : {f1_train_score_gs} and model test acore : {f1_test_score_gs}")

                #checking if the model is underfitting 
                logging.info(f"checking for underfitting condition : {self.model_trainer_config.expected_score} expected score")
                if f1_test_score_gs < self.model_trainer_config.expected_score:
                    raise Exception(f" Model performing poor, \
                        expected score {self.model_trainer_config.expected_score} :model_score {f1_test_score_gs}")
            
                #checking if model overfitted
                diff = abs(f1_train_score_gs-f1_test_score_gs)
                logging.info(f"difference between train ans test score: {diff}")

                logging.info(f"checking for overfitting condition:{self.model_trainer_config.overfitting_threshold} threshold ")
                if diff > self.model_trainer_config.overfitting_threshold:
                    raise Exception(f"Model Overfitted as difference in train and test :{diff} \
                        more than threshold {self.model_trainer_config.overfitting_threshold}")

                #saving model
                logging.info(f"saving model")
                utils.save_object(file_path=self.model_trainer_config.model_path, obj=model_gs)

                #Artifact

                model_trainer_artifact = artifact_entity.ModelTrainerArtifact(
                    model_path=self.model_trainer_config.model_path,
                    f1_train_score=f1_train_score_gs,
                    f1_test_score=f1_test_score_gs)

                logging.info(f"model trainer artifact: {model_trainer_artifact}")
                return model_trainer_artifact

            else:
                logging.info(f" model is best model with given parameters")
                logging.info(f" model train score : {f1_train_score} and model test acore : {f1_test_score}")

                #checking if the model is underfitting 
                logging.info(f"checking for underfitting condition : {self.model_trainer_config.expected_score} expected score")
                if f1_test_score < self.model_trainer_config.expected_score:
                    raise Exception(f" Model performing poor, \
                        expected score {self.model_trainer_config.expected_score} :model_score {f1_test_score}")
            
                #checking if model overfitted
                diff = abs(f1_train_score-f1_test_score)
                logging.info(f"difference between train ans test score: {diff}")

                logging.info(f"checking for overfitting condition:{self.model_trainer_config.overfitting_threshold} threshold ")
                if diff > self.model_trainer_config.overfitting_threshold:
                    raise Exception(f"Model Overfitted as difference in train and test :{diff} \
                        more than threshold {self.model_trainer_config.overfitting_threshold}")

                #saving model
                logging.info(f"saving model")
                utils.save_object(file_path=self.model_trainer_config.model_path, obj=model)

                #Artifact

                model_trainer_artifact = artifact_entity.ModelTrainerArtifact(
                    model_path=self.model_trainer_config.model_path,
                    f1_train_score=f1_train_score,
                    f1_test_score=f1_test_score)

                logging.info(f"model trainer artifact: {model_trainer_artifact}")
                return model_trainer_artifact
        except Exception as e:
            raise ThyDetectException(e, sys)

