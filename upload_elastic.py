from elasticsearch import Elasticsearch
from elasticsearch import helpers
import pandas as pd
import csv
import os
import json


def upload_Books(book_index):
    #x = pd.read_csv("BX-Books.csv")
    es = Elasticsearch(['https://localhost:9200/'], ssl_assert_fingerprint="090d01c3894ea9e5de046d07100f6af34287c8c69955fdc1e9b394ea61b6695f",basic_auth=("elastic", "Xh7dY1eDHw6YqrsH+h+0"))

    books = pd.read_csv("BX-Books.csv")
    ratings=pd.read_csv("BX-Book-Ratings.csv")
    users = pd.read_csv("BX-Users.csv")
    #print(len(books),len(ratings),len(users))

    #merge dataframes
    res=books.merge(ratings , on='isbn' ,how="inner")

    #convert to csv format

    #res.to_csv("/home/spetz/Desktop/information_Retrieval/final_book.csv")



    json_str = books.to_json(orient='records')

    json_records = json.loads(json_str)

    index_name = str(book_index)
    es.indices.delete(index=index_name, ignore=[400, 404])
    es.indices.create(index=index_name, ignore=400)   
    action_list = []
    for row in json_records:
        record ={
            '_op_type': 'index',
            '_index': index_name,
            '_source': row
        }
        action_list.append(record)
    helpers.bulk(es, action_list)
    if es.indices.exists(index=book_index):
        print(es.indices.get_alias())

def upload_ratings(rating_index):
    #x = pd.read_csv("BX-Books.csv")
    es = Elasticsearch(['https://localhost:9200/'], ssl_assert_fingerprint="090d01c3894ea9e5de046d07100f6af34287c8c69955fdc1e9b394ea61b6695f",basic_auth=("elastic", "Xh7dY1eDHw6YqrsH+h+0"))

    books = pd.read_csv("BX-Books.csv")
    ratings=pd.read_csv("BX-Book-Ratings.csv")
    users = pd.read_csv("BX-Users.csv")
    #print(len(books),len(ratings),len(users))

    #merge dataframes
    res=books.merge(ratings , on='isbn' ,how="inner")

    #convert to csv format

    #res.to_csv("/home/spetz/Desktop/information_Retrieval/final_book.csv")



    json_str = ratings.to_json(orient='records')

    json_records = json.loads(json_str)

    index_name = str(rating_index)
    es.indices.delete(index=index_name, ignore=[400, 404])
    es.indices.create(index=index_name, ignore=400)   
    action_list = []
    for row in json_records:
        record ={
            '_op_type': 'index',
            '_index': index_name,
            '_source': row
        }
        action_list.append(record)
    helpers.bulk(es, action_list)
    if es.indices.exists(index=rating_index):
        print(es.indices.get_alias())        


        