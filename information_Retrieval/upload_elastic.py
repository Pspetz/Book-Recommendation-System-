from elasticsearch import Elasticsearch
from elasticsearch import helpers
import pandas as pd
import csv
import os

def upload_Books(book_index):
    #x = pd.read_csv("BX-Books.csv")
    es = Elasticsearch(['http://localhost:9200/'], verify_certs=True)

    
   # es.indices.create(index='query3', ignore=400)

    df = pd.read_csv('BX-Books.csv', delimiter=',', encoding="utf-8", skipinitialspace=True)
        # print(df)
    if es.indices.exists(index=book_index,ignore=400):
        pass
    else:
        with open('BX-Books.csv') as f:
            reader = csv.DictReader(f)
            helpers.bulk(es, reader, index=book_index)

    if es.indices.exists(index=book_index):
        print(es.indices.get_alias())

def upload_Users(user_index):
    es = Elasticsearch(['http://localhost:9200/'] , verify_certs=True)

    if es.indices.exists(index = user_index , ignore=400):
        pass
    else:
        with open('BX-Users.csv') as f:
            reader = csv.DictReader(f)
            helpers.bulk(es, reader, index=user_index)

    if es.indices.exists(index=user_index):
        print(es.indices.get_alias())


def upload_Ratings(ratings_index):
    es = Elasticsearch(['http://localhost:9200/'] , verify_certs=True)

    if es.indices.exists(index = ratings_index , ignore=400):
        pass
    else:
        with open('BX-Book-Ratings.csv') as f:
            reader = csv.DictReader(f)
            helpers.bulk(es, reader, index=ratings_index)

    if es.indices.exists(index=ratings_index):
        print(es.indices.get_alias())

#es.options(ignore_status=[400,404]).indices.delete(index='test-index')


