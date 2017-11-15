'''
Sam Congdon and Bradley White
CSCI 440: Data Science Application
November 15, 2017

This class interacts directly with the database to perform queries for analysis
'''

import sqlite3


class DBManager():
    def __init__(self, db_path):
        # Open the DB connection at the path
        self.conn = sqlite3.connect(db_path)
        # Initialize a cursor object to perform queries
        self.cursor = self.conn.cursor()

    # Returns a list of tuples, where a tuple contains the attributes in the SELECT statement and each row is a tuple
    def perform_query(self, sql_query):
        return self.cursor.execute(sql_query).fetchall()

    # Close the connection cleanly
    def close_connection(self):
        self.conn.close()
