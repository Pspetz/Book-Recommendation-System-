from upload_elastic import * 
from search import *
import sys
from costum_search import *

es = Elasticsearch(['https://localhost:9200/'], ssl_assert_fingerprint="5d62e0cf4920a91659a61a9ad0ac417c02161538a931ec881f16cc842ce88b3d",basic_auth=("elastic", "HYyElLskPkbcjmpiIskE"))


def menu():

    print("----------welcome to Main menu--------- \n")
    print("1------>insert data------\n")
    print("2------>search data------\n")
    print("3------>costum search data------\n")
    print("4------>exit------\n")


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

        else:
            sys.exit("thanks for using our Cluster")



if __name__ == '__main__':
    main()