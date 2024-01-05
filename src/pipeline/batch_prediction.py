from src.exception import ThyDetectException
from src.logger import logging
import os,sys
from src.Prediction import ModelResolver
from src.utils import load_object,make_df,convert_column_dtype,trans,inverse_trans
from src.config import col_names
import pandas as pd
from datetime import datetime
import numpy as np 

PREDICTION_FILE_PATH = "prediction"

def start_batch_prediction(input_file_path):
    try:
        prediction_dir = os.path.join(os.getcwd(),PREDICTION_FILE_PATH)
        os.makedirs(prediction_dir,exist_ok=True)

        model_resolver = ModelResolver(model_registry="saved_models")

        # Loading objects
        transformer = load_object(file_path=model_resolver.get_latest_transformer_path())
        model = load_object(file_path=model_resolver.get_latest_model_path())
        label_encoder = load_object(file_path=model_resolver.get_latest_encoder_path())

        #preparing dataset
        df=pd.read_csv(input_file_path)
        df.replace(to_replace='?',value=np.NAN,inplace=True)
        df=make_df(df=df, col_names=col_names)
        df=convert_column_dtype(df=df)

        object_cols = df.select_dtypes(include='object').columns.tolist()

        #encoding
        df = trans(cols=object_cols, encoder=label_encoder, df=df)
        target = df[col_names[0]]

        input_feature_name = list(transformer.feature_names_in_)
        input_arr = transformer.transform(df[input_feature_name])
        prediction = model.predict(input_arr)

        #inverse transform
        cat_prediction = inverse_trans(col=col_names[0], encoder=label_encoder, y_pred=prediction)

        df['prediction'] = prediction
        df['cat_prediction'] = cat_prediction

        prediction_file_name = os.path.basename(input_file_path).replace(".csv",f"{datetime.now().strftime('%d%m%Y__%H%M%s')}.csv")
        prediction_file_path = os.path.join(prediction_dir,prediction_file_name)
        df.to_csv(prediction_file_path,index=False,header=True)
        return prediction_file_path

    except Exception as e:
        raise ThyDetectException(e, sys)