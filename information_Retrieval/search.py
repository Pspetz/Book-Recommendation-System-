from elasticsearch import Elasticsearch
from elasticsearch import helpers
import pandas as pd
import csv
import os
import requests, json

#id = input('give id:')


def search_query(): 
    query = {
        "query": {
            "match": {
                'user.id': int(input('give book id:'))
            }
        }
    } 
    return query



requestHeaders = {'content-type': 'application/json'}

def searcher(url):
    es = Elasticsearch(['http://localhost:9200/'], verify_certs=True)
    #es.indices.refresh(index="k") 
    my_query = search_query() #INSERT id_book.
    results = requests.get(url, data=json.dumps(my_query), headers=requestHeaders) #REQUEST FOR THE SEARCH RESULTS.
    print(results)


searcher('http://localhost:9200/')