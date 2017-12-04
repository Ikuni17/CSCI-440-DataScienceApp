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
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold



# Get the results from the DB for a specific question
def query_db(db, question_num):
    # Get the correct query for the question
    if question_num == 1:
        query = 'SELECT DISTINCT K.Tconst, K.Revenue, G.Genre ' \
                'FROM KAGGLE K, Genre G ' \
                'WHERE K.Tconst = G.Tconst ' \
                'AND K.Revenue IS NOT NULL'
    elif question_num == 2:
        query = 'SELECT DISTINCT Avg_rating, Num_votes ' \
                'FROM RATINGS ' \
                'WHERE Num_votes > 0'
    elif question_num == 3:
        query = 'SELECT DISTINCT E.Season_Num, R.Avg_rating ' \
                'FROM EPISODE E, RATINGS R ' \
                'WHERE E.Econst = R.Tconst ' \
                'AND E.Season_Num IS NOT NULL ' \
                'AND E.Season_Num < 50 ' \
                'AND R.Avg_rating IS NOT NULL'
    elif question_num == 4:
        query = 'SELECT DISTINCT Revenue, Start_year, Runtime, Face_number, ' \
                'FB_likes, Rank, Meta_score ' \
                'FROM IMDB I, KAGGLE K ' \
                'WHERE I.Tconst = K.Tconst ' \
                'AND Revenue IS NOT NULL ' \
                'AND Start_year IS NOT NULL ' \
                'AND Runtime IS NOT NULL ' \
                'AND Face_number IS NOT NULL ' \
                'AND Budget IS NOT NULL ' \
                'AND FB_likes IS NOT NULL ' \
                'AND Rank IS NOT NULL ' \
                'AND Meta_score IS NOT NULL'

    elif question_num == 5:
        query = "SELECT DISTINCT K.Revenue, K.Budget, K.Content_rating, R.Avg_rating " \
                "FROM KAGGLE K, RATINGS R " \
                "WHERE K.Tconst = R.Tconst " \
                "AND K.Revenue IS NOT NULL " \
                "AND K.Budget IS NOT NULL " \
                "AND K.Content_rating IS NOT NULL"

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
    # plt.legend()
    # plt.show()

    leverage_plot = sm.graphics.influence_plot(reg, size=15)
    leverage_plot.savefig('leverage.png')

    exog = sm.graphics.plot_regress_exog(reg)
    exog.savefig('exog.png')


# Perform analysis specific to question 1: Mean Revenue by Genre
def perform_1(db):
    result = query_db(db, 1).fetchall()
    result_dict = {}

    # Denormalize the genres into a dictionary with the structure: {Tconst: [[Revenue], [Genres]]}
    for row in result:
        if row[0] in result_dict:
            if row[2] not in result_dict[row[0]][1]:
                result_dict[row[0]][1].append(row[2])
        else:
            result_dict[row[0]] = [[row[1]], [row[2]]]

    # Create another dictionary with genres as the key and a list of revenues for that genre as the value
    genre_dict = {}
    for k, v in result_dict.items():
        genres = frozenset(v[1])

        if genres in genre_dict:
            genre_dict[genres].append(v[0][0])
        else:
            genre_dict[genres] = [v[0][0]]

    print(genre_dict)
    # print(result_dict)

    # f, p = scipy.stats.f_oneway(genre_dict)


# Perform analysis specific to question 2: Linear Regression Num Votes and Rating
def perform_2(db):
    ''' This method analyzes how well Logistic Regression can predict a Title_type of a production.
        The prediction is performed using 4 attributes: Is_adult, R.Avg_rating, Start_year, and Runtime.
        The results of the predictions are recorded in three heat maps, plotted against combination of
        the latter three attributes. These heatmaps show patterns in how logistic regressions was able
        to classify the data in response to the attributes. '''

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
    # 291663 rows
    result = query_db(db, 3)
    # Convert to numpy array
    temp_vector = np.fromiter(result.fetchall(), 'i4,f')
    # Split into two vectors

    num_seasons = temp_vector['f0']
    rating = temp_vector['f1']

    plt.figure(figsize=(14, 7))

    # retrieve the attributes of line fit with linear regression
    slope, intercept, r, p, std_error = scipy.stats.linregress(rating, num_seasons)
    plt.scatter(rating, num_seasons, label='Data')
    plt.plot(rating, intercept + slope * rating, 'r', label="Fit, r={0}".format(r))

    print(
        "Slope: {0}\nIntercept: {1}\nr: {2}\np-value: {3}\nstd. error: {4}\n".format(slope, intercept, r, p, std_error))
    plt.legend()
    plt.ylabel('Number of Seasons')
    plt.title('Linear Regression for Number of Seasons vs. Show Rating')

    plt.xlabel('Show Rating')
    plt.savefig('Results3.pdf')
    plt.show()
    '''


# Perform analysis specific to question 4: Predict Revenue
# Run PCA then use neural net to predict revenue
def perform_4(db):
    # 642 rows, not sure if title should be considered
    result = query_db(db, 4).fetchall()
    labels = ['Revenue', 'Start_year', 'Runtime', 'Face_number', 'FB_likes', 'Rank', 'Meta_score']

    # f_regression is an alternate feature selection method, however tuning revealed mutual_info_regression to
    # perform better on the selected data. Thus, f_regression is not used in the final application
    f_regress = False

    # split into predicted value revenue and input variables data
    revenue = np.array([x[0] for x in result])
    data = [x[1:] for x in result]

    plt.figure(figsize=(14, 7))
    plt.grid(True)
    axes = plt.gca()
    #axes.set_xlim([0, 7])
    #axes.set_ylim([0, 250000])
    number_of_tests = 100
    trend = {'x': [1,2,3,4,5,6], 'y': [0,0,0,0,0,0]}

    for j in range(5):
        output = {'x': [], 'y': []}
        for i in reversed(range(1, 7)):
            data = [(x[1:]) for x in result]
            pca = PCA(n_components=i)
            pca.fit(data)
            data = pca.transform(data)
            df = pd.DataFrame(data)
            kf = KFold(n_splits=15)
            pca_output = []

            for train_index, test_index in kf.split(data):
                # train the network
                clf = MLPRegressor(alpha=0.01, hidden_layer_sizes=(100,), max_iter=50000, early_stopping=False, batch_size=100,
                                   activation='relu', solver='adam', verbose=False,
                                   learning_rate_init=0.001, learning_rate='adaptive', tol=0.000000000000000000001)

                clf.fit(data[train_index], revenue[train_index])

                pred_revenue = clf.predict(data[test_index])  # predict network output given input data
                target_revenue = revenue[test_index]
                error = [(pred_revenue[x] - target_revenue[x])**2 for x in range(len(pred_revenue))]
                error = sum(error) / len(error)
                #print("predicted rev = {}".format(pred_revenue))
                #print('target rev = {}'.format(target_revenue))
                #print('error = {}'.format(error))

                pca_output += [error]
                plt.scatter(i, error, color=colors[j])

            output['y'] += [sum(pca_output) / len(pca_output)]
            output['x'] += [i]
            #print('network outputs = {}'.format(pca_output))
            #print('avg error of folds = {}'.format(sum(pca_output) / len(pca_output)))
            #print(output)


    trend['y'] = [(trend['y'][x] / 5) + 150000 for x in range(len(trend['y']))] + [150000]
    plt.plot(trend['x'], trend['y'], color='m')

    plt.xlabel('Number of Components')
    plt.ylabel('Average Mean Square Error of Networks')
    if f_regress:
        plt.savefig('Results4 with f_regression.png')
    else:
        plt.savefig('Results4.png')
    #plt.show()

# Perform analysis specific to question 5: Predict Revenue
# do multiple linear regression to predict revenue using
# sam don't work on 5
def perform_5(db):
    result = query_db(db, 5).fetchall()

    revenue = np.array([x[0] for x in result])
    data = [x[1:] for x in result]

    ratings = np.unique([x[1] for x in data])
    encoded = {ratings[i]: '00000' for i in range(len(ratings))}

    df = pd.DataFrame({'rating': ratings})
    print(pd.get_dummies(df))

    clf = lm.LinearRegression()
    clf.fit(data, revenue)


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

    perform_1(db)

    # Close the database connection cleanly
    db.close_connection()


if __name__ == '__main__':
    main()
