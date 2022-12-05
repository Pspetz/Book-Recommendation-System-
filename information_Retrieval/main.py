from upload_elastic import * 



def main():
    index1 = input(str('give a index_book string value:'))
    index2 = input(str('give a index_user string value:'))
    index3 = input(str('give a index_rating string value:'))

    if (index1 != index2 and index1 != index3 and index2 != index3):

        upload_Books(index1)
        upload_Users(index2)
        upload_Ratings(index3)



    else:
        print("choose other index value")
        main()


if __name__ == '__main__':
    main()