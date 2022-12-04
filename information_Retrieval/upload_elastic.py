from elasticsearch import Elasticsearch
from elasticsearch import helpers
import pandas as pd
import csv
import os

def uploader():
    #x = pd.read_csv("BX-Books.csv")
    es = Elasticsearch(['http://localhost:9200/'], verify_certs=True)

    
    es.indices.create(index='query3', ignore=400)

    df = pd.read_csv('BX-Books.csv', delimiter=',', encoding="utf-8", skipinitialspace=True)
        # print(df)

    if es.indices.exists(index="query3"):
        pass
    else:
        with open('BX-Books.csv') as f:
            reader = csv.DictReader(f)
            helpers.bulk(es, reader, index='query3')

    #if es.indices.exists(index='query2'):
        #print(es.indices.get_alias())

    # check data is in there, and structure in there
    #es.search(body={"query": {"match_all": {}}}, index = 'query1')
    #es.indices.get_mapping(index = 'query1')


uploader()