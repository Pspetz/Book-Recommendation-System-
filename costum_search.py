from elasticsearch import Elasticsearch
from elasticsearch import helpers
import pandas as pd


url = "https://localhost:9200/books/_search"
requestHeaders = {'content-type': 'application/json'}

# create new custom metric function for score, rating,similarity 

def metric1(x):
    return x[0]*0.5 + x[1]*0.25 + x[2]*0.25

es = Elasticsearch(['https://localhost:9200/'], ssl_assert_fingerprint="5d62e0cf4920a91659a61a9ad0ac417c02161538a931ec881f16cc842ce88b3d",basic_auth=("elastic", "HYyElLskPkbcjmpiIskE"))

def costum_search():
    books= pd.DataFrame()

        #-----Elasticsearch metric-----
    book = input("What is the book you want to search for?\n")
    query = {
                "match": {
                    "book_title":book
                }
            }

    results = es.search(index='books',query=query)
    books = pd.json_normalize(results['hits']['hits'])
    
    if(books.empty):
        print("cannot find any book, try again!")
        costum_search()
    
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

        resp = es.search(index="ratings" , query=query)

        resp=pd.json_normalize(resp['hits']['hits'])
        resp

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
            costum_search()



