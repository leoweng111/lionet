"""
This script is for some basic operation on MongoDB.
"""
import os
import pandas as pd
from typing import Union, List
from pymongo import UpdateOne

from .mongoconfig import client
from lwpackage.lwutils import log, PREP_PATH


def update_data(database: str,
                collection: str,
                df: pd.DataFrame,
                method: str = 'insert_many',
                filter_column: Union[str, list] = None):
    """
    General method for updaating data.
    Attention: update_many method is not used because it can not update multiple records with different filters once.

    :param database: database
    :param collection: collection
    :param df: dataframe with one row represents one record
    :param method:
        1. update_one: Using update_one, robust but slow method. This will add new data and overwrite old data.
        2. bulk_write_update: Using UpdateOne + bulk_write. This will add new data and overwrite old data.
        3. insert_many: Using insert_many, This will add new data but will NOT overwrite old data.
    :param filter_column: filter_column for update data.
    :return:
    """
    if method == 'insert_many':
        insert_data(database=database,
                    collection=collection,
                    df=df)
        return

    if not filter_column:
        filter_column = ['time', 'instrument_id']
    for col in filter_column:
        assert col in df.columns, f'df does not contain column {col}, df columns: {df.columns}.'

    if method == 'update_one':
        for _, row in df.iterrows():
            mongo_operator = {}
            for col in filter_column:
                mongo_operator[col] = row[col]

            update_one_data(database=database,
                            collection=collection,
                            mongo_operator=mongo_operator,
                            data=row.to_dict(),
                            upsert=True)

    elif method == 'bulk_write_update':
        bulk_write_update_data(database=database,
                               collection=collection,
                               df=df,
                               filter_column=filter_column)
    else:
        raise ValueError("Only supports method 'update_one', 'bulk_write_update' or 'insert_many'.")


def bulk_write_update_data(database: str,
                           collection: str,
                           df: pd.DataFrame,
                           filter_column: Union[str, list]):
    """
    Update stock price data from the given df using UpdateOne and bulk_wtite.
        1. If data from df already exists in collecion, overwrite the old data with the data from df.
        2. If data from df does not exist in collection, insert new data from df into the collection.

    :param database: database
    :param collection: collection
    :param df: data
    :param filter_column:
    :return:
    BulkWriteResult 对象包含了执行批量写入操作后的结果。您可以查看以下属性来获取有关更新的信息：
    acknowledged：一个布尔值，指示批量写入是否已被确认。如果为 True，则表示写入操作已被确认；如果为 False，则表示写入操作未被确认。
        如果写入操作未被确认，那么其他属性的值将无法确定。
    inserted_count：插入的文档数量。表示成功插入到集合中的文档数。
    matched_count：匹配到的文档数量。用于更新操作，表示满足筛选条件的文档数。
    modified_count：修改的文档数量。用于更新操作，表示实际被修改的文档数。
    deleted_count：删除的文档数量。用于删除操作，表示成功删除的文档数。
    upserted_count：插入或更新的文档数量。用于更新操作中的 upsert 操作，表示新插入的文档数。
    upserted_ids：一个字典，将操作索引映射到插入的文档的 _id。用于更新操作中的 upsert 操作。
    """
    assert all(df.notna().all())
    assert not any(df.duplicated())
    for col in ['time', 'instrument_id', 'open', 'high', 'low', 'close', 'volume']:
        assert col in df.columns
    df = df[['time', 'instrument_id', 'open', 'high', 'low', 'close', 'volume']].copy()
    if isinstance(filter_column, str):
        filter_column = [filter_column]

    try:
        update_operations = []
        for _, row in df.iterrows():
            filter_dc = {}
            for col in filter_column:
                filter_dc[col] = row[col]
            update_row = {"$set": row.to_dict()}
            operation = UpdateOne(filter_dc, update_row, upsert=True)
        update_operations.append(operation)

        result = client[database][collection].bulk_write(update_operations)

        print(f'Inserted {result.inserted_count} documents into {database}.{collection}.')
        print(f'Matched {result.matched_count} documents for update on {database}.{collection}.')
        print(f'Modified {result.modified_count} documents in {database}.{collection}.')
        print(f'Upserted {result.upserted_count} documents into {database}.{collection}.')

        log.info(f'Inserted {result.inserted_count} documents into {database}.{collection}.')
        log.info(f'Matched {result.matched_count} documents for update on {database}.{collection}.')
        log.info(f'Modified {result.modified_count} documents in {database}.{collection}.')
        log.info(f'Upserted {result.upserted_count} documents into {database}.{collection}.')

    except Exception as e:
        print(f'Error occurs: {e}')
        log.info(f'Error occurs: {e}')


def insert_data(database: str,
                collection: str,
                df: pd.DataFrame):
    """
        Insert data using insert_many. If data already exists in database, will NOT overwrite the old data.

    :param df: data
    :param database: database
    :param collection: collection
    :return:
    """
    assert all(df.notna().all())
    assert not any(df.duplicated())
    try:
        data = df.to_dict(orient='records')
        result = client[database][collection].insert_many(data)

        print(f'Successfully insert {len(result.inserted_ids)} records into {database}.{collection}.')
        log.info(f'Successfully insert {len(result.inserted_ids)} records into {database}.{collection}.')

    except Exception as e:
        print(f'insertMany Error ouucrs: {e}')
        log.info(f'insertMany Error ouucrs: {e}')


def update_one_data(database: str,
                    collection: str,
                    mongo_operator: dict,
                    data: dict,
                    upsert: bool = True):
    """
    Updata one record into database. It will overwrite old data and add new data.

    :param database: database
    :param collection: collection
    :param mongo_operator: mongo_operator
    :param data: dict-format data
    :param upsert: If True, then will overwrite old data.
    :return:
    """
    try:
        col = client[database][collection]
        col.update_one(mongo_operator, {'$set': data}, upsert=upsert)

    except Exception as e:
        print(f'insertMany Error ouucrs: {e}')
        log.info(f'insertMany Error ouucrs: {e}')


def get_data(database: str,
             collection: str,
             mongo_operator: Union[dict, None] = None,
             idx: bool = False):
    """
    Get data from database.

    :param database: database
    :param collection: database
    :param mongo_operator: mongo_operator
    :param idx: data with idx or not, default to False.
    :return:
    """
    col = client[database][collection]
    if not idx:
        df = pd.DataFrame(col.find(mongo_operator, {"_id": False}))
    else:
        df = pd.DataFrame(col.find(mongo_operator))

    return df


def delete_data(database: str,
                collection: str,
                mongo_operator: Union[dict, None] = None):
    """
    Delete data from datbase. Attention! This can be very dangerous.
    """
    if not mongo_operator:
        client[database][collection].delete_many({})
        log.warning(f'Delete all data in {database}.{collection}.')
    else:
        client[database][collection].delete_many(mongo_operator)
        log.warning(f'Delete required data in {database}.{collection}.')


def check_exists(database: Union[str, list],
                 collection: Union[str, list, None] = None,
                 mongo_operator: Union[str, None] = None):
    """
    Check data exists with optional filters. If given multi databases, only check databases; if given one database and
    many collections, only check collections, if one database and one collection and given mongo filter, check following
    mongo filter.

    :param database: database
    :param collection: collection
    :param mongo_operator: mongo_operator
    :return: None
    """
    if isinstance(database, list) and len(database) > 1:
        assert not collection, f'When specify multi databases, should not specify any collections.'
        assert not mongo_operator, f'When specify multi databases, should not specify mongo operator.'

        return check_database_exists(database=database)

    if isinstance(database, list) and len(database) == 1:
        database = database[0]

    if isinstance(collection, list) and len(database) > 1:
        assert not mongo_operator, f'When specify multi collections, should not specify mongo operator.'
        return check_collection_exists

    if isinstance(collection, list) and len(collection) == 1:
        collection = collection[0]
    return check_data_exists(database=database,
                             collection=collection,
                             mongo_operator=mongo_operator)


def check_database_exists(database: Union[str, List]):
    """
    Check database exists in client or not.
    :param database: database
    :return: None
    """

    if isinstance(database, str):
        database = [database]
    db_not_exist = []
    for db in database:
        if db not in client.list_database_names():
            db_not_exist.append(db)
    if len(db_not_exist) > 0:
        log.info(f'Database in {db_not_exist} do not exist.')
        return False
    return True


def check_collection_exists(database: str,
                            collection: Union[str, List]):
    """
    Check collections exist in a specify database of client.
    :param database: database
    :param collection: collection
    :return: None
    """
    if isinstance(collection, str):
        collection = [collection]
    col_not_exist = []
    for col in collection:
        if col not in client[database].list_collection_names():
            col_not_exist.append(col)
    if len(col_not_exist) > 0:
        log.info(f'Collections in {col_not_exist} do not exist.')
        return False
    return True


def check_data_exists(database: str,
                      collection: str,
                      mongo_operator: str):
    """
    Check collections exist in a specify collection of a specific database of client.

    :param database: database
    :param collection: collection
    :param mongo_operator: mongo_operator
    :return: None
    """
    return len(client[database][collection].find(mongo_operator)) > 0


def get_collection_storage_size(database: str = 'stock',
                                collection: str = 'price_1min'):
    """
    Gets the storage size of a collection in a MongoDB database.

    :param database: database
    :param collection: collection
    :return: float: Collection storage size in megabytes (MB).
    """
    try:
        db = client[database]
        collection_stats = db.command("collstats", collection)
        storage_size_bytes = collection_stats["size"]
        storage_size_mb = storage_size_bytes / (1024 * 1024)  # Convert to MB
        print(f"Collection size in MB: {storage_size_mb}")

        return storage_size_mb

    except Exception as e:
        print(f"Error: {e}")
        return None


def drop_duplicated_data(database: str,
                         collection: str,
                         subset: Union[list, None] = None):
    """
    Drop duplicated data of a colleciton. Attention! This can be very dangerous.

    :param database: drop_duplicates
    :param collection: collection
    :param subset: unique key, subset param of dataframe.drop_duplicates
    :return: None
    """

    df = get_data(database=database, collection=collection)

    # save df for robustness
    file_name = f'data_from_{database}_{collection}' + 'pkl'
    file_path = os.path.join(PREP_PATH, file_name)
    df.to_pickle(file_path)
    num_before = len(df)
    if subset is not None:
        df = df.drop_duplicates(subset=subset)
    else:
        df = df.drop_duplicates()
    num_after = len(df)
    delete_data(database=database, collection=collection)
    insert_data(database=database,
                collection=collection,
                df=df)
    num_drop = num_before - num_after

    print(f'Successfully drop {num_drop} duplicated data of {database} {collection} with unique key {subset}')
    log.info(f'Successfully drop {num_drop} duplicated data of {database} {collection} with unique key {subset}')
