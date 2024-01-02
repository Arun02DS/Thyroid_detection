import pymongo
import pandas as pd
import json
from dataclasses import dataclass
import os

"""
This file will make a connection with mongo db client.

"""

@dataclass
class EnvironmentVariable:
    mongo_db_url= os.getenv("MONGO_DB_URL")

env_var = EnvironmentVariable()
# Provide the mongodb localhost url to connect python to mongodb.
mongo_client = pymongo.MongoClient(env_var.mongo_db_url)

col_names = ['Diagnosis','sex']
pram_grid={
    'learning_rate':[0,1e-5, 1e-2, 0.001, 0.005, 0.01, 0.05,0.1,1,5,50, 100],
    'max_depth':range(3,10,2),
    'min_child_weight':range(1,6,2),
    #'gamma':[i/10.0 for i in range(0,5)],
    #'subsample':[i/100.0 for i in range(75,90,5)],
    #'colsample_bytree':[i/100.0 for i in range(75,90,5)],
    #'reg_alpha':[1e-5, 1e-2, 0.1,0, 0.001, 0.005, 0.01, 0.05,1, 100]
}