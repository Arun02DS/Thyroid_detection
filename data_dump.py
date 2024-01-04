import pymongo
import pandas as pd
import json

"""
This file upload/dump the csv format file on mongo database.

"""


# Provide the mongodb localhost url to connect python to mongodb.
client = pymongo.MongoClient("mongodb://localhost:27017/neurolabDB")

DATA_FILE_PATH="/config/workspace/thyroid_data_.csv"
DATABASE_NAME="thyroid"
COLLECTION_NAME="thydetect"

if __name__=="__main__":
    # Reading csv file and creating a dataframe
    df = pd.read_csv(DATA_FILE_PATH)

    #Converting dataframe into json format
    df.reset_index(drop=True,inplace=True)

    json_records = list(json.loads(df.T.to_json()).values())
    

    # After converting to json push records into mongo db
    client[DATABASE_NAME][COLLECTION_NAME].insert_many(json_records)
    print(f"****Records uploaded in Mongodb****\nrows and columns : {df.shape},\n1st row record in json format: \n{json_records[0]}")
