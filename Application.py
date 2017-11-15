'''
Sam Congdon and Bradley White
CSCI 440: Data Science Application
November 15, 2017
'''

import DB_Manager
import sqlite3

def main():
    brad_path = "C:\\IMDB\\D3 Python Script\\imdb.db"
    #sam_path = "imdb.db"
    db = DB_Manager.DBManager(brad_path)
    print(db.perform_query('SELECT * FROM IMDB'))
    db.close_connection()


if __name__ == '__main__':
    main()