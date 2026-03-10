"""
This script is for the basic configuration of MongoDB.
"""
from utils.path import MONGODB_PASSWORD, USER_NAME
from pymongo import MongoClient
from utils.logging import log

mongo_url = f'mongodb://{USER_NAME}:{MONGODB_PASSWORD}@127.0.0.1:27017'

client = MongoClient(mongo_url)
log.info(f'Successfully connect to MongoClient {USER_NAME}.')

futures_db = client['futures']
