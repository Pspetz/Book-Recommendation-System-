
from elasticsearch import Elasticsearch
import tensorflow as tf
from tensorflow import keras
import keras
import re
from nltk.tokenize import word_tokenize
import numpy as np
import pandas as pd
import string
from keras import losses
from keras.models import Sequential
from keras.layers import Dense, Softmax
from sklearn.model_selection import train_test_split
from numpy import array
from numpy import asarray
import nltk
nltk.download('stopwords')
nltk.download('punkt')
import matplotlib.pyplot as plt
from numpy import zeros
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer # Used for stemming
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.losses import SparseCategoricalCrossentropy # Loss function being used

csv  = pd.read_csv("final_cluster.csv")
csv = csv[['cluster' , 'uid' ,'location','age' , 'isbn' ,'book_title' ,'book_author','summary','rating']]



# Delete all the rows from dataframe which have rating = 0
cluster = csv.where(csv['rating'] != 0).dropna()
#cluster1_new = csv_cluster1.where(csv_cluster1['rating'] != 0).dropna()
cluster

avg_rating = cluster.groupby(['isbn'])['rating'].mean().reset_index(name='rating')
#avg_rating1 = cluster1_new.groupby(['isbn'])['rating'].mean().reset_index(name='rating')

avg_rating

# Create user rating dataframe ,whice contains all info about each movie rating
new_df = cluster[['cluster','summary','uid','rating','isbn']]


# Text-cleaning function
def clean_data(csv):
    #remove with regex all punctuation
    text = re.sub('[^A-Za-z0-9]+', ' ', csv)
    #lowercase
    text = text.lower()
    return text


new_df['Clean'] = new_df['summary'].apply(clean_data)

stop_words = set(stopwords.words('english'))
new_df['removestopwords'] = new_df['Clean'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))



#SnowballStemer(trasnform the word to shorter word with the same root meaning)
stemmer = SnowballStemmer("english")

#create new series('stemmed')
new_df['Stemmed'] = new_df['removestopwords'].apply(lambda x: ' '.join([stemmer.stem(word) for word in str(x).split()]))
new_df =new_df.drop(columns=['Clean'])
new_df =new_df.drop(columns=['removestopwords'])



#split dataset to cluster0 and cluster1 
cluster0 = new_df[new_df['cluster'] == 0 ]
cluster1 = new_df[new_df['cluster'] == 1 ]

#CLUSTER 0

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
# Tokenize the summary texts
# weight matrix of one embedding for each unique word in summary 
summaries0 = cluster0["Stemmed"].copy()

# Tokenize the summary texts
token = Tokenizer()
token.fit_on_texts(summaries0)
#len of each unique word in summary 
vocab_size0 = len(token.word_index) + 1 
texts0 = token.texts_to_sequences(summaries0) # encode to integer each word 
texts0 = pad_sequences(texts0, padding='post')
texts0

#CLUSTER 1
summaries1 = cluster1["Stemmed"].copy()
token = Tokenizer()
token.fit_on_texts(summaries1)
vocab_size1 = len(token.word_index) + 1 
texts1 = token.texts_to_sequences(summaries1) 
texts1 = pad_sequences(texts1, padding='post')
print(texts0,texts1)

#sequence for each word
embeddings_index = dict()
f = open('glove.6B.100d.txt', encoding="utf8")
for line in f:
    values = line.split()
    word = values[0]
    coefs = asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Loaded %s word vectors.' % len(embeddings_index))

#Cluster 0
# Create a weight matrix of one embedding for each unique word in summary texts
embedding_matrix0 = zeros((vocab_size0, 100))
for word, i in token.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix0[i] = embedding_vector

#CLUSTER 1
embedding_matrix1 = zeros((vocab_size1, 100))
for word, i in token.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix1[i] = embedding_vector

#CLUSTER 0 
textTrain, textTest, ratingTrain, ratingTest = train_test_split(texts0, cluster0['rating'], test_size=0.33)
ratingTest
#CLUSTER 1
#CLUSTER 0 
textTrain1, textTest1, ratingTrain1, ratingTest1 = train_test_split(texts1, cluster1['rating'], test_size=0.33)
ratingTest1

print(vocab_size0,texts0.shape[1])

def eval_model(vocab_size,embedding_matrix,texts,textTrain,ratingTrain,textTest,ratingTest):
#input_dim -> megethos tou leksilogiou
#output_dim -> akeraios arithmos tou dimension of embedding
#input_length -> mikos akolouthias eisodou
    input_length = texts.shape[1]
#here Y output is intiger so we use sparse_categorical_Crossentropy
    model = keras.Sequential()
    model.add(keras.layers.Embedding(input_dim=vocab_size, output_dim=100, weights=[embedding_matrix], input_length=texts.shape[1], trainable=False))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(512, activation='relu'))
    model.add(keras.layers.Dense(256, activation='relu' ))
    model.add(keras.layers.Dense(128, activation='relu' ))
    model.add(keras.layers.Dense(11, activation='softmax'))

    model.summary()


# Compile the model
    model.compile(loss=SparseCategoricalCrossentropy(from_logits = True), optimizer='adam', metrics=['accuracy'])
# Fit the model
    history = model.fit(textTrain, ratingTrain, epochs=100, batch_size=50, validation_split = 0.2,callbacks=[tf.keras.callbacks.EarlyStopping(monitor="loss", patience=5)])

    loss, accuracy = model.evaluate(textTest, ratingTest) # Get the loss and accuracy based on the tests

    return model,history

cluster0_res,cluster0_his = eval_model(vocab_size0,embedding_matrix0,texts0,textTrain,ratingTrain,textTest,ratingTest)

cluster1_res,cluster1_his = eval_model(vocab_size1,embedding_matrix1,texts1,textTrain1,ratingTrain1,textTest1,ratingTest1)

def create_model_plots(history0,history1):
        plt.figure(0)
        plt.subplot(2, 2, 1)
        plt.plot(history0.history['accuracy'], label='Accuracy (train)')
        plt.plot(history0.history['val_accuracy'], label='Accuracy (test)')
        plt.title("Cluster 0 ")
        plt.ylabel("Accuracy")
        plt.xlabel("Epochs")
        plt.legend()

        plt.subplot(2, 2, 2)
        plt.plot(history1.history['accuracy'], label='Accuracy (train)')
        plt.plot(history1.history['val_accuracy'], label='Accuracy (test)')
        plt.title("Cluster 1 ")
        plt.ylabel("Accuracy")
        plt.xlabel("Epochs")
        plt.tight_layout()
        plt.show()



create_model_plots(cluster0_his,cluster1_his)

#cluster0-cluster1
new_cluster0 = csv[csv['cluster'] ==0]
new_cluster1=csv[csv['cluster'] == 1]

cluster0_null = new_cluster0[new_cluster0['rating'] == 0]
cluster1_null = new_cluster1[new_cluster1['rating'] == 0]

def cluster_clean_data(csv,texts):
    csv
    csv = csv[['summary' ,'rating']]
    csv
    x=texts[0]
    y = csv['rating']

    #clean data
    csv['summary'] =csv['summary'].apply(clean_data)
    csv['removestopwords'] = csv['summary'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))



    #create new series('stemmed')
    csv['Stemmed'] = csv['removestopwords'].apply(lambda x: ' '.join([stemmer.stem(word) for word in str(x).split()]))
    csv =csv.drop(columns=['removestopwords'])

    return csv

cluster0_clean = cluster_clean_data(cluster0_null,texts0)
cluster1_clean = cluster_clean_data(cluster1_null,texts1)


# Tokenize the summary texts
# weight matrix of one embedding for each unique word in summary 
summaries = cluster0_clean["summary"].copy()
input_length0 = texts0.shape[1]
# Tokenize the summary texts
token = Tokenizer()
token.fit_on_texts(summaries)
#len of each unique word in summary 
vocab_size0 = len(token.word_index) + 1 
new_texts0 = token.texts_to_sequences(summaries) # encode to integer each word 
new_texts0 = pad_sequences(new_texts0, padding='post',maxlen=input_length0)



# Tokenize the summary texts
# weight matrix of one embedding for each unique word in summary 
summaries = cluster1_clean["summary"].copy()
input_length1 = texts1.shape[1]
# Tokenize the summary texts
token = Tokenizer()
token.fit_on_texts(summaries)
#len of each unique word in summary 
vocab_size1 = len(token.word_index) + 1 
new_texts1 = token.texts_to_sequences(summaries) # encode to integer each word 
new_texts1 = pad_sequences(new_texts1, padding='post',maxlen=input_length1)

new_df0 = pd.DataFrame()
predict0_x=cluster0_res.predict(new_texts0) 
new_df0=np.argmax(predict0_x,axis=1)


new_df1 = pd.DataFrame()
predict1_x=cluster1_res.predict(new_texts1) 
new_df1=np.argmax(predict1_x,axis=1)


#CONNECT CLUSTERS BACK TO SAME DATAFRAME #PREDICTED RATING
clust0=cluster0_null.drop(columns=['rating'])
clust0['rating']=new_df0

clust1=cluster1_null.drop(columns=['rating'])
clust1['rating']=new_df1



csv_new=csv[csv['rating'] != 0 ]
#csv_new =csv_new.append(clust0)

fcsv=clust0.append(csv_new)
final_Csv=clust1.append(fcsv)

final_Csv_books = final_Csv[['isbn',"book_title","book_author",'summary']]
final_Csv_ratings=final_Csv[['location','isbn','rating']]


from upload_elastic import helpers
import csv
import json
es = Elasticsearch(['https://localhost:9200/'], ssl_assert_fingerprint="090d01c3894ea9e5de046d07100f6af34287c8c69955fdc1e9b394ea61b6695f",basic_auth=("elastic", "Xh7dY1eDHw6YqrsH+h+0"))

index = input(str('give a string value for booking upload:'))

def upload_Books(book_index):
    json_str = final_Csv_books.to_json(orient='records')

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
upload_Books(index)

index1 = input(str('give a string value for rating upload:'))

def upload_ratings(rating_index):
    
    json_str = final_Csv_ratings.to_json(orient='records')

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
upload_ratings(index1)



