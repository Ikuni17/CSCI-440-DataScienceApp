'''
Sam Congdon and Bradley White
CSCI 440: Data Science Application
November 15, 2017
'''

import sqlite3

class DBManager():
    def __init__(self, db_path):
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()

    def perform_query(self, sql_query):
        return self.cursor.execute(sql_query).fetchall()

    def close_connection(self):
        self.conn.close()