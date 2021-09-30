import pandas as pd
from pymongo import MongoClient, collection

data = pd.read_csv('testTwitterClusterData.csv')

client = MongoClient("mongodb://127.0.0.1:27017/")

db = client['twitter']
collection = db['tweets']

while True:
    data.reset_index(drop=True)

    data_dict = data.to_dict("records")
    collection.insert_many(data_dict)