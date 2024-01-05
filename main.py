from src.pipeline.training_pipeline import start_training_pipeline
from src.pipeline.batch_prediction import start_batch_prediction

input_file_path = "/config/workspace/thyroid_data_.csv"

if __name__=="__main__":
     try:
          # Starting training pipeline.
          #start_training_pipeline()
          output=start_batch_prediction(input_file_path=input_file_path)
          print(output)

     except Exception as e:
          print(e)

