from upload_elastic import * 
from search import *
import sys
from costum_search import *
from kmeans_search import *

es = Elasticsearch(['https://localhost:9200/'], ssl_assert_fingerprint="090d01c3894ea9e5de046d07100f6af34287c8c69955fdc1e9b394ea61b6695f",basic_auth=("elastic", "Xh7dY1eDHw6YqrsH+h+0"))


def menu():

    print("----------welcome to Main menu--------- \n")
    print("1------>insert data------\n")
    print("2------>search data------\n")
    print("3------>costum search data------\n")
    print("4------>kmeans_search data------\n")
    print("5------>exit------")


    choise = input(str("choose a selection: \n"))

    return choise

def main():
    epilogi =menu()
    condition = False 
    condition1 = True

    while condition == False:

        if(epilogi == "1"):


            index1 = input(str('give a string value for booking upload:'))
            index2 = input(str('give a string value for rating upload:'))

            if (len(index1)!=0):

                upload_Books(index1)
                upload_ratings(index2)

                print("do you want to do something else?")
                ing=str(input('yes or no:'))

                if(ing=='y' or ing=='yes'):
                    main()

                else: 
                    sys.exit("thanks for using our Cluster")



        elif(epilogi=='2'):

            url = "https://localhost:9200/books/_search"
            url1 = "https://localhost:9200/ratings/_search"

            x=searcher(url,url1)
            #final_list = x[['book_title','rating']]
            #print(x)
            print("do you want to do something else?")
            ing1=str(input('yes or no:'))

            if(ing1=='y' or ing1=='yes'):
                main()

            else: 
                sys.exit("thanks for using our Cluster")

        elif(epilogi=='3'):
            y=costum_search()

            print(" want you to search for something else?")
            ing2=str(input('yes or no'))
            if(ing2=='y' or ing2=='yes'):
                main()

            else:
                sys.exit("thanks for using our Cluster")

        elif(epilogi =='4'):
            up = input(str('want you to upload kmeans data first?'))
            if(up == 'yes' or up =='y'):
                print('first, we need to upload the new data..')
                kmeans_index1 = input(str('give a string value for booking upload:'))
                kmeans_index2 = input(str('give a string value for rating upload:'))
                if (len(kmeans_index1)!=0 and len(kmeans_index2)):
                    upload_Books1(kmeans_index1)
                    upload_ratings1(kmeans_index2)

                    print("searching for...?")

                    search_kmeans=str(input('yes or no:'))

                    if(search_kmeans=='y' or search_kmeans=='yes'):
                        x=searcher1()

                    else: 
                        sys.exit("thanks for using our Cluster")
            else:
                print("searching for...?")

                search_kmeans=str(input('yes or no:'))

                if(search_kmeans=='y' or search_kmeans=='yes'):
                    x=searcher1()

                else:
                    sys.exit("thanks for using our Cluster")

        
        else:
            sys.exit("thanks for using our Cluster")



if __name__ == '__main__':
    main()