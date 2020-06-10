from pymongo import MongoClient
from Settings import *
from datetime import datetime


# Clear all data in db
def clear_db(db):
    db.delete_many({})


# Add data to db
def to_db(load, db_collection):
    db_collection.insert_one(load, db_collection)


def mongo_init():
    client = MongoClient(MONGO_SOURCE)
    db = client[DATABASE_NAME]
    db_collection = db.nn_outputs
    if REWRITE_DB:
        clear_db(db_collection)
    return db_collection


def upload_result(res, db_collection):
    # Data for loading to DB
    timestamp = datetime.now().timestamp()  # Unix epoch time
    cam_id = CAM_ID

    result = []
    for r in res:
        result.append(int(r))

    # Prepare load format to insert to db
    load = {'timestamp': int(timestamp),
            'cam_id': cam_id,
            'result': result}

    # DB writing
    to_db(load, db_collection)

