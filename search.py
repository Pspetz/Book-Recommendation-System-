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
                'book_title': str(input('give summary:'))
            }
        }
    } 
    return query
 
def search_query1(): 
    query1 = {
        "query": {
            "match": {
                'uid': int(input('give user uid:'))
            }
        }
    } 
    return query1
 

requestHeaders = {'content-type': 'application/json'}
book1_list=list()
book_list = list()
rating_list=list()
def searcher(url,url1):
    es = Elasticsearch(['https://localhost:9200/'], ssl_assert_fingerprint="5d62e0cf4920a91659a61a9ad0ac417c02161538a931ec881f16cc842ce88b3d",basic_auth=("elastic", "HYyElLskPkbcjmpiIskE"))
    #es.indices.refresh(index="k") 
    my_query = search_query() #INSERT id_book.
    my_query1 = search_query1()
    results = requests.get(url, data=json.dumps(my_query), verify=False,auth=("elastic", "HYyElLskPkbcjmpiIskE"),headers=requestHeaders) #REQUEST FOR RESULTS
    results1 = requests.get(url1, data=json.dumps(my_query1), verify=False,auth=("elastic", "HYyElLskPkbcjmpiIskE"),headers=requestHeaders) #REQUEST FOR RESULTS

    #print(results)
    data =results.json()
    #print(data)
    #print( results.text)
    #print(data['hits']['hits'][0]['_source'])

    for i in range(len(data['hits']['hits'])):
        book_list.append(data['hits']['hits'][i]['_source'])
    
    x=pd.DataFrame(book_list)
    print(book1_list)
    books = x[['book_title','isbn']]

    
    print("ta vivlia pou vrethikan kai tairiazoun ston titlo pou dwsate einai\n" ,books)



     #print(results)
    data1 =results1.json()
   

    for i in range(len(data1['hits']['hits'])):
        rating_list.append(data1['hits']['hits'][i]['_source'])


    x1=pd.DataFrame(rating_list)
    ratings = x1[['isbn','rating']]
    print("O xristis exei vathmologisi tis sugkekrimenes tenies:\n",ratings)


    final = ratings.merge(books, on="isbn", how = 'inner')
    final = pd.DataFrame(final)
    final = final.drop(columns=['isbn'])

    if len(final) == 0:
        print("den uparxei diathesimi vathmologia me auto to user id")
    else:
        print('o xristis pou anazitisate exei vathmologisi tin antistoixi tainia pou anazitisate:\n',final)

    print("score for BM25 similarity metric \n")
    search_score = data['hits']['hits']
    for hit in search_score:
        print(hit['_score'],'\t',hit['_source']['book_title'])




    


    





