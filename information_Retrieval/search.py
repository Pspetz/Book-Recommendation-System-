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
                'isbn': str(input('give book id:'))
            }
        }
    } 
    return query



requestHeaders = {'content-type': 'application/json'}

book_list = list() 
def searcher(url):
    es = Elasticsearch(['http://localhost:9200/'], verify_certs=True)
    #es.indices.refresh(index="k") 
    my_query = search_query() #INSERT id_book.
    results = requests.get(url, data=json.dumps(my_query), headers=requestHeaders) #REQUEST FOR RESULTS
    #print(results)
    data =results.json()
    #print( results.text)
    print(data['hits']['hits'][0])

searcher('http://localhost:9200/book/_search')