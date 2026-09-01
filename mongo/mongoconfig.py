"""
This script is for the basic configuration of MongoDB.
"""
from utils.params import MONGODB_PASSWORD, USER_NAME
from pymongo import MongoClient
from utils.logging import log

mongo_url = f'mongodb://{USER_NAME}:{MONGODB_PASSWORD}@127.0.0.1:27017'

# Short server-selection timeout so that sync pymongo calls in async FastAPI
# routes fail fast instead of blocking the event loop for the pymongo default
# (30s) when MongoDB is briefly unavailable or saturated.
client = MongoClient(
    mongo_url,
    serverSelectionTimeoutMS=5000,
    connectTimeoutMS=5000,
    socketTimeoutMS=10000,
    maxPoolSize=50,
)
log.info(f'Successfully connect to MongoClient {USER_NAME}.')

futures_db = client['futures']
