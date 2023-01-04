import pandas as pd
import matplotlib.pyplot as plt 
import string
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import plotly.express as exp
from sklearn import preprocessing
import tqdm
import time

def kmeans_upload():
    user_csv = pd.read_csv("BX-Users.csv")
    user_rating = pd.read_csv("BX-Book-Ratings.csv")
    book = pd.read_csv("BX-Books.csv")
    len(user_csv)
    user_csv
    user_csv['location']=user_csv['location'].str.replace(" " , "")
    user_csv['location']=user_csv['location'].str.replace("," , "")

    le = preprocessing.LabelEncoder()
    for column_name in user_csv.columns:
        if user_csv['location'].dtype == object:
            user_csv['location'] = le.fit_transform(user_csv['location'])
        else:
            pass

    user_csv_final = pd.read_csv("BX-Users.csv")
    user_csv_final

    res=book.merge(user_rating , on='isbn' ,how="inner")
    res

    user_csv.info()
    user_csv.isnull().sum()

    user_csv['age'] = user_csv['age'].fillna(int(user_csv['age'].mean()))
    user_csv_final['age'] = user_csv_final['age'].fillna(int(user_csv_final['age'].mean()))
    user_csv_final

    #user_csv['location'] = user_csv['location'].str.replace(" " , "")
    user_csv

    scalar=MinMaxScaler()
    scaled_data = scalar.fit_transform(user_csv)
    scaled_data

    scaled_df = pd.DataFrame(data = scaled_data,columns=user_csv.columns[0:])

    scaled_df=scaled_df.drop(columns=['uid'])
    scaled_df

    data = scaled_df
    # Calculate sum of squared distances
    ssd = []
    K = range(1,10)
    for k in K:
        km = KMeans(n_clusters=k)
        km = km.fit(data)
        ssd.append(km.inertia_) 
    plt.figure(figsize=(10,6))
    plt.plot(K, ssd, 'bx-')
    plt.xlabel('k')
    plt.ylabel('ssd')
    plt.title('Elbow Method For Optimal k')
    plt.show()

    kmean = KMeans(n_clusters=2)
    kmean.fit(data)
    pred = kmean.labels_

    pred
    exp.scatter(data_frame= user_csv,x = 'location',y = 'age',color=kmean.labels_)

    user_csv_final
    kmeans_csv = user_csv.merge(user_csv_final, how ='inner' , on ='uid')
    kmeans_csv = kmeans_csv.drop(columns=['location_x','age_x'])
    kmeans_csv = kmeans_csv.rename(columns={"location_y":'location' , 'age_y':"age"})
    kmeans_csv

    cluster_list = pd.DataFrame()
    cluster_list['cluster'] = kmean.labels_

    cluster_list
    con = pd.concat([cluster_list,kmeans_csv],axis=1, ignore_index=False)
    con

    merged_data = con.merge(res,how="inner" , on='uid')

    merged_data=merged_data.drop(columns=['publisher','category','year_of_publication'])
    merged_data

    #cluster_map = merged_data.groupby(['cluster', 'location'])['rating'].mean().reset_index().sort_values(['cluster','rating'],ascending=False)

    cluster0 = merged_data[(merged_data["cluster"]==0)]
    cluster1 = merged_data[(merged_data['cluster']==1)]
    cluster0

    #NOW WE WANT TO PREDICT RATING WITH SAME CLUSTER USERS 
    #cluster0
    #avg_book_rating = cluster0.groupby('rating')['book_title'].mean()


    ##cluster0

    cluster_map0 = cluster0.groupby(['book_title'])['rating'].mean().reset_index().sort_values(['rating'],ascending=False)
    cluster_map1 = cluster1.groupby(['book_title'])['rating'].mean().reset_index().sort_values(['rating'],ascending=False)

    cluster_map0= cluster_map0[cluster_map0['rating'] != 0]
    cluster_map1= cluster_map1[cluster_map1['rating'] != 0]

    #cluster0['rating'].replace(to_replace = 0, value = cluster_map['rating'].mean(), inplace=True)
    cluster0

    check_list=list()
    for i in range(len(cluster0)):
        if cluster0['rating'].iloc[i] == 0:
            x=cluster0[['cluster','uid','location','age','isbn','book_title','book_author','summary','rating']].iloc[i]
            check_list.append(x)
        else:
            continue

    csv_null=pd.DataFrame(check_list)



    check_list1=list()
    for i in range(len(cluster1)):
        if cluster1['rating'].iloc[i] == 0:
            y=cluster1[['cluster','uid','location','age','isbn','book_title','book_author','summary','rating']].iloc[i]
            check_list1.append(y)
        else:
            continue

    csv_null1=pd.DataFrame(check_list1)

    csv_null
    csv_null1

    for i in range(len(csv_null)):
        for j in range(len(cluster_map0)):
            if (csv_null['book_title'].iloc[i] == cluster_map0['book_title'].iloc[j] and csv_null['rating'].iloc[i] ==0):
                name_book=csv_null['book_title'].iloc[i]
                new_value=cluster_map0['rating'].iloc[j]
                old_value = csv_null['rating'].iloc[i]
                #cluster0['rating'].replace(old_value,new_value)
                csv_null.loc[csv_null['book_title'] == name_book, 'rating'] = int(new_value)


    for i in range(len(csv_null1)):
        for j in range(len(cluster_map1)):
            if (csv_null1['book_title'].iloc[i] == cluster_map1['book_title'].iloc[j] and csv_null1['rating'].iloc[i] ==0):
                name_book1=csv_null1['book_title'].iloc[i]
                new_value1=cluster_map1['rating'].iloc[j]
                old_value1 = csv_null1['rating'].iloc[i]
                #cluster0['rating'].replace(old_value,new_value)
                csv_null1.loc[csv_null1['book_title'] == name_book1, 'rating'] = int(new_value1)

    cluster0_without_null= cluster0[cluster0['rating'] != 0]
    cluster1_without_null = cluster1[cluster1['rating'] != 0]

    print(csv_null,csv_null1)

    final_cluster0 = cluster0_without_null.append(csv_null)
    final_cluster1 = cluster1_without_null.append(csv_null1)

    final_cluster = final_cluster0.append(final_cluster1)

    final_cluster.to_csv('/home/spetz/Desktop/Information_Retrieval-main/final_cluster.csv')

    final_cluster 
    Kmeans_BX_Ratings = final_cluster[['uid' , 'isbn' , 'rating']]
    '''for i in range(len(Kmeans_BX_Ratings)):
        if Kmeans_BX_Ratings['rating'].iloc[i] == 0:
            Kmeans_BX_Ratings['rating']=Kmeans_BX_Ratings['rating'].replace(0 , 1)
        
        else:
            continue'''

    Kmeans_BX_Ratings

    Kmeans_BX_Ratings.to_csv('/home/spetz/Desktop/Information_Retrieval-main/Kmeans_ratings.csv')

    Kmeans_BX_Books = final_cluster.drop(columns = ['cluster' , 'uid' , 'age' ,'rating'])
    Kmeans_BX_Books

    Kmeans_BX_Ratings.to_csv('/home/spetz/Desktop/Information_Retrieval-main/Kmeans_ratings.csv')
    Kmeans_BX_Books.to_csv('/home/spetz/Desktop/Information_Retrieval-main/Kmeans_books.csv')

