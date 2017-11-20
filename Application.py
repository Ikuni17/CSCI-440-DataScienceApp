'''
Sam Congdon and Bradley White
CSCI 440: Data Science Application
November 15, 2017
'''

import DB_Manager
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import platform
import seaborn
import scipy.stats
import sklearn.linear_model as lm
import statistics as stats
import statsmodels
import statsmodels.api as sm
import statsmodels.formula.api as smf


# Get the results from the DB for a specific question
def query_db(db, question_num):
    # Get the correct query for the question
    if question_num == 1:
        pass
    elif question_num == 2:
        query = 'SELECT Avg_rating, Num_votes FROM RATINGS WHERE Num_votes > 20000'
    elif question_num == 3:
        pass
    elif question_num == 4:
        pass
    elif question_num == 5:
        pass

    return db.perform_query(query)


def diagnostic_plots(vector):
    reg = smf.ols('f0 ~ f1', data=vector).fit()
    qq_plot = sm.qqplot(reg.resid, line='r')
    qq_plot.savefig('qq.png')

    stdres = pd.DataFrame(reg.resid_pearson)
    plt.plot(stdres, 'o', ls='None')
    l = plt.axhline(y=0, color='r')
    plt.ylabel('Standardized Residual')
    plt.xlabel('Observation Number')
    #plt.legend()
    #plt.show()

    leverage_plot = sm.graphics.influence_plot(reg, size=15)
    leverage_plot.savefig('leverage.png')

    exog = sm.graphics.plot_regress_exog(reg)
    exog.savefig('exog.png')

# Perform analysis specific to question 1: Mean Revenue by Genre
def perform_1(db):
    pass


# Perform analysis specific to question 2: Linear Regression Num Votes and Rating
def perform_2(db):
    # 754224 rows
    result = query_db(db, 2)
    # Convert to numpy array
    temp_vector = np.fromiter(result.fetchall(), 'f,i4')
    diagnostic_plots(temp_vector)
    # Split into two vectors
    rating = temp_vector['f0']
    num_votes = temp_vector['f1']
    # Log Transform number of votes to linearize the relationship
    log_num_votes = np.log(num_votes)

    '''use_transform = True

    plt.style.use('seaborn')
    plt.figure(figsize=(15.5, 9.5), dpi=100)

    if use_transform:
        slope, intercept, r, p, std_error = scipy.stats.linregress(log_num_votes, rating)
        plt.scatter(log_num_votes, rating, label='Data')
        plt.plot(log_num_votes, intercept + slope * log_num_votes, 'r', label="Fit, r={0}".format(r))
    else:
        slope, intercept, r, p, std_error = scipy.stats.linregress(num_votes, rating)
        plt.scatter(num_votes, rating, label='Data')
        plt.plot(num_votes, intercept + slope * num_votes, 'r', label="Fit, r={0}".format(r))

    print(
        "Slope: {0}\nIntercept: {1}\nr: {2}\np-value: {3}\nstd. error: {4}\n".format(slope, intercept, r, p, std_error))
    plt.legend()
    plt.ylabel('Avg Rating')
    plt.title('Linear Regression for Rating vs. Number of Votes')

    if use_transform:
        plt.xlabel('Log - Number of Votes')
        plt.savefig('Results\\2-Transformed.png')
    else:
        plt.xlabel('Number of Votes')
        plt.savefig('Results\\2.png')
    plt.show()'''


# Perform analysis specific to question 3: Num Seasons and Show Rating
def perform_3(db):
    pass


# Perform analysis specific to question 4: Predict Revenue
def perform_4(db):
    pass


# Perform analysis specific to question 5: Predict Remake Rating
def perform_5(db):
    pass


def main():
    # Determine the path to the DB based on who is running this application
    if platform.system() is 'Windows':
        # Brad's Path
        path = "C:\\IMDB\\D3 Python Script\\imdb.db"
    else:
        # Sam's Path
        path = "/Users/Samuel/PycharmProjects/CSCI-440-DataScienceApp/imdb.db"

    # Create a database manager based on the path
    db = DB_Manager.DBManager(path)

    #perform_2(db)

    # Close the database connection cleanly
    db.close_connection()


if __name__ == '__main__':
    main()
