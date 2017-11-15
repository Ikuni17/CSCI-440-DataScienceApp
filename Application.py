'''
Sam Congdon and Bradley White
CSCI 440: Data Science Application
November 15, 2017
'''

import DB_Manager

def main():
    brad_path = "C:\\IMDB\\D3 Python Script\\imdb.db"
    #sam_path = "imdb.db"
    db = DB_Manager.DBManager(brad_path)
    query = 'SELECT * FROM IMDB WHERE Runtime > 1000'
    print(type(db.perform_query(query)))

    db.close_connection()


if __name__ == '__main__':
    main()