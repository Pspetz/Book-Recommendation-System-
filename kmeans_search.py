from elasticsearch import Elasticsearch
from elasticsearch import helpers
import pandas as pd
import csv
import os
import json
import requests, json


def upload_Books1(book_index):
    #x = pd.read_csv("BX-Books.csv")
    es = Elasticsearch(['https://localhost:9200/'], ssl_assert_fingerprint="090d01c3894ea9e5de046d07100f6af34287c8c69955fdc1e9b394ea61b6695f",basic_auth=("elastic", "Xh7dY1eDHw6YqrsH+h+0"))

    kmeans_books = pd.read_csv("Kmeans_books.csv")
    kmeans_ratings=pd.read_csv("Kmeans_ratings.csv")
    #print(len(books),len(ratings),len(users))

    #merge dataframes
    res=kmeans_books.merge(kmeans_ratings , on='isbn' ,how="inner")

    #convert to csv format

    #res.to_csv("/home/spetz/Desktop/information_Retrieval/final_book.csv")



    json_str = kmeans_books.to_json(orient='records')

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

def upload_ratings1(rating_index):
    #x = pd.read_csv("BX-Books.csv")
    es = Elasticsearch(['https://localhost:9200/'], ssl_assert_fingerprint="090d01c3894ea9e5de046d07100f6af34287c8c69955fdc1e9b394ea61b6695f",basic_auth=("elastic", "Xh7dY1eDHw6YqrsH+h+0"))

    kmeans_books = pd.read_csv("Kmeans_books.csv")
    kmeans_ratings=pd.read_csv("Kmeans_ratings.csv")
    #print(len(books),len(ratings),len(users))

    #merge dataframes
    res=kmeans_books.merge(kmeans_ratings , on='isbn' ,how="inner")

    #convert to csv format

    #res.to_csv("/home/spetz/Desktop/information_Retrieval/final_book.csv")



    json_str = kmeans_ratings.to_json(orient='records')

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

#id = input('give id:')



url = "https://localhost:9200/books/_search"
requestHeaders = {'content-type': 'application/json'}

# create new custom metric function for score, rating,similarity 

def metric1(x):
    return x[0]*0.5 + x[1]*0.25 + x[2]*0.25

es = Elasticsearch(['https://localhost:9200/'], ssl_assert_fingerprint="090d01c3894ea9e5de046d07100f6af34287c8c69955fdc1e9b394ea61b6695f",basic_auth=("elastic", "Xh7dY1eDHw6YqrsH+h+0"))

def searcher1():
    books= pd.DataFrame()

        #-----Elasticsearch metric-----
    book = input("What is the book you want to search for?\n")
    query = {
                "match": {
                    "book_title":book
                }
            }

    results = es.search(index='kmeansbooks',query=query)
    books = pd.json_normalize(results['hits']['hits'])
    
    if(books.empty):
        print("cannot find any book, try again!")
        searcher1()
    
    else:

    #Keep stats
        scores = books["_score"]
        isbn = books["_source.isbn"]

        isbn

        #books = books.set_index("_source.isbn")

        books

        #Search user ratings:
        user_id = int(input("give your uid?\n"))

        query = {
            "match":{
                "uid":user_id
            }

        }
        
        
        resp = es.search(index="kmeansratings" , query=query)

        resp=pd.json_normalize(resp['hits']['hits'])
        resp

        if(resp.empty):
            print("cannot find any uid, try again!")
            searcher1()
        else:
            search_user_rating = [0]*len(scores)
            for i in range(len(isbn)):
                for j in range(len(resp['_source.isbn'])):
                    if isbn[i] ==resp['_source.isbn'][j]:
                        search_user_rating[i] = resp['_source.isbn'][j]

            #check for non empty list
            if [x for x in search_user_rating if x]:

                #B25 SIMILARITY METRIC( DEFAULT ELASTIC METRIC)
                print(books[["_score","_source.book_title"]])

                #for i in range(len(books)):
                    #books['_score'][i] = metric([scores[i],float(search_user_rating[i])])

                #new_books = books.sort_values(by=["_score"],ascending=False)

                #new_books=new_books[['_score',"_source.book_title"]]

                #new_books

                books

                from sklearn.feature_extraction.text import CountVectorizer
                from sklearn.metrics.pairwise import cosine_similarity

                bookss=books[['_source.summary','_source.book_title']]

                def comb_features(row):
                    return row['_source.summary'] + " " + row["_source.book_title"]

                bookss['new_feature'] = bookss.apply(comb_features,axis=1)
                bookss

                cv=CountVectorizer()
                final_matrix = cv.fit_transform(bookss["new_feature"])
                final_matrix = final_matrix.toarray()
                final_matrix

                cosine_similarity = cosine_similarity(final_matrix)
                cosine_similarity

                df_new =bookss.pivot_table(index='_source.book_title' ,columns = '_source.summary')

                x=list()
                for i in range(len(books)):
                    x.append(i)

                x=pd.DataFrame(x)
                con = pd.concat([x,books],axis=1, ignore_index=False)
                index_val=list()

                for i in range(len(books)):
                    in_val=con[con['_source.book_title']== books['_source.book_title'][i]].index.values
                    index_val.append(in_val)
                index_val

                import numpy as np
                for i in range(len(index_val)):
                        similar_books =list(enumerate(cosine_similarity[index_val[i]]))
                        books["_score"][i] = metric1([scores[i], float(search_user_rating[i]), float(similar_books[0][1][i])])

                books

                books = books.sort_values(by=['_score'], ascending=False)

                print(books[["_score", "_source.book_title"]])
        
            else:
                print("den uparxei antistoixeia metaksi book kai user id,dokimaste ksana!")
                searcher1()



