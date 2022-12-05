from elasticsearch import Elasticsearch
from elasticsearch import helpers
import pandas as pd
import csv
import os

def uploader(index):
    #x = pd.read_csv("BX-Books.csv")
    es = Elasticsearch(['http://localhost:9200/'], verify_certs=True)

    
   # es.indices.create(index='query3', ignore=400)

    df = pd.read_csv('BX-Books.csv', delimiter=',', encoding="utf-8", skipinitialspace=True)
        # print(df)
    if es.indices.exists(index=index,ignore=400):
        pass
    else:
        with open('BX-Books.csv') as f:
            reader = csv.DictReader(f)
            helpers.bulk(es, reader, index=index)

    if es.indices.exists(index=index):
        print(es.indices.get_alias())



#es.options(ignore_status=[400,404]).indices.delete(index='test-index')


