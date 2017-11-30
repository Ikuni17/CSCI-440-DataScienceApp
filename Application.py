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
                'AND E.Season_Num > 0 ' \
                'AND R.Avg_rating IS NOT NULL ' \
                'AND R.Avg_rating > 0 ' \
                'AND R.Avg_rating < 10'
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

    # exog = sm.graphics.plot_regress_exog(reg)
    # exog.savefig('exog.png')


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
    # 754224 rows
    result = query_db(db, 2)
    # Convert to numpy array
    temp_vector = np.fromiter(result.fetchall(), 'f,i4')
    diagnostic_plots(temp_vector)
    # Split into two vectors
    rating = temp_vector['f0']
    num_votes = temp_vector['f1']

    '''use_transform = True

    plt.style.use('seaborn')
    plt.figure(figsize=(15.5, 9.5), dpi=100)

    if use_transform:
        # Log Transform number of votes to linearize the relationship
        log_num_votes = np.log(num_votes)
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
    diagnostic_plots(temp_vector)
    # Split into two vectors

    num_seasons = temp_vector['f0']
    #num_seasons = np.log(num_seasons)
    rating = temp_vector['f1']

    df = pd.DataFrame(rating)
    '''model = lm.LogisticRegression()
    model.fit(df, num_seasons)

    plt.scatter(rating, num_seasons)
    plt.plot(rating, model.predict(df), color='r')

    plt.ylabel('Number of Seasons')
    plt.xlabel('Show Rating')
    plt.savefig('Results3.png')
    plt.show()


    # check the accuracy on the training set
    print("Model Score is: {}".format(model.score(df, num_seasons)))
    

    print(rating)'''
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


# Perform analysis specific to question 4: Predict Revenue
# Run PCA then use neural net to predict revenue
def perform_4(db):
    # 642 rows, not sure if title should be considered
    result = query_db(db, 4).fetchall()

    # split into predicted value revenue and input variables data
    colors = {0: 'k', 1: 'b', 2: 'g', 3: 'r', 4: 'c', 5: 'y', 6: 'm'}
    revenue = np.array([x[0] for x in result])
    data = [x[1:] for x in result]

    plt.figure(figsize=(14, 7))
    plt.grid(True)
    axes = plt.gca()
    axes.set_xlim([0, 7])
    axes.set_ylim([0, 250000])
    trend = {'x': [1, 2, 3, 4, 5, 6], 'y': [0, 0, 0, 0, 0, 0]}

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
                clf = MLPRegressor(alpha=0.01, hidden_layer_sizes=(100,), max_iter=50000, early_stopping=False,
                                   batch_size=100,
                                   activation='relu', solver='adam', verbose=False,
                                   learning_rate_init=0.001, learning_rate='adaptive', tol=0.000000000000000000001)

                clf.fit(data[train_index], revenue[train_index])

                pred_revenue = clf.predict(data[test_index])  # predict network output given input data
                target_revenue = revenue[test_index]
                error = [(pred_revenue[x] - target_revenue[x]) ** 2 for x in range(len(pred_revenue))]
                error = sum(error) / len(error)
                # print("predicted rev = {}".format(pred_revenue))
                # print('target rev = {}'.format(target_revenue))
                # print('error = {}'.format(error))

                pca_output += [error]
                plt.scatter(i, error, color=colors[j])

            output['y'] += [sum(pca_output) / len(pca_output)]
            output['x'] += [i]
            # print('network outputs = {}'.format(pca_output))
            # print('avg error of folds = {}'.format(sum(pca_output) / len(pca_output)))
            # print(output)

        trend['y'] = [trend['y'][x] + (output['y'][x] - output['y'][x + 1]) for x in range(len(output['y']) - 1)]
        plt.plot(output['x'], output['y'], color=colors[j])

    trend['y'] = [(trend['y'][x] / 5) + 150000 for x in range(len(trend['y']))] + [150000]
    plt.plot(trend['x'], trend['y'], color='m')

    plt.xlabel('Number of Components')
    plt.ylabel('Average Mean Square Error of Networks')
    plt.legend()
    plt.savefig('Results5Linear.png')
    plt.show()
    return

    '''
    #best lines come from PCA of 5,6,4
    # loop to compare the linear regression over the different number of PCA components used
    for i in reversed(range(1, 7)):
        data = [(x[1:]) for x in result]
        pca = PCA(n_components=i)
        pca.fit(data)
        data = pca.transform(data)
        df = pd.DataFrame(data)

        clf = MLPRegressor(alpha=0.01, hidden_layer_sizes=(100,), max_iter=50000, early_stopping=False, batch_size=100,
                       activation='relu', solver='adam', verbose=True,
                       learning_rate_init=0.001, learning_rate='adaptive', tol=0.000000000000000000001)
        clf.fit(df, revenue)
        pred_revenue = clf.predict(data)  # predict network output given input data
        revenue = np.array(revenue)
        slope, intercept, r, p, std_error = scipy.stats.linregress(revenue, pred_revenue)
        plt.plot(revenue, intercept + slope * revenue, colors[i], label="Fit, r={}, slope = {}".format(r, slope))

    plt.xlabel('Acutal Revenue (in Millions)')
    plt.ylabel('Predicted Revenue (in Millions)')
    plt.legend()
    plt.savefig('Results5Linear.png')
    plt.show()
    return
    '''

    # create and train network
    clf = MLPRegressor(alpha=0.01, hidden_layer_sizes=(10,), max_iter=50000, early_stopping=False, batch_size=100,
                       activation='relu', solver='adam', verbose=True,
                       learning_rate_init=0.001, learning_rate='adaptive', tol=0.000000000000000000001)
    clf.fit(df, revenue)

    pred_revenue = clf.predict(data)  # predict network output given input data
    plt.scatter(revenue, pred_revenue)  # plot network output vs target output
    plt.xlabel('Acutal Revenue (in Millions)')
    plt.ylabel('Predicted Revenue (in Millions)')

    # create a linear regression line, ideal fit has slope == 1
    revenue = np.array(revenue)
    slope, intercept, r, p, std_error = scipy.stats.linregress(revenue, pred_revenue)
    plt.plot(revenue, intercept + slope * revenue, 'r', label="Fit, r={}, slope = {}".format(r, slope))

    plt.legend()
    plt.savefig('Results5Linear.png')
    plt.show()

    '''
    pca = PCA()
    pca.fit(result)
    print(pca.explained_variance_)
    print(pca.explained_variance_ratio_)

    pca = PCA()
    pca.fit(result)
    print(pca.explained_variance_ratio_)
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')
    plt.savefig('Results4.png')
    plt.show()
    '''
    # pca = PCA(n_components=7)
    # pca.fit(result)
    # print(pca.explained_variance_)

    # X_pca = pca.transform(result)
    # print("original shape:   ", len(result[0]))
    # print("transformed shape:", X_pca.shape)


# Perform analysis specific to question 5: Predict Revenue
# do multiple linear regression to predict revenue using
def perform_5(db):
    result = query_db(db, 5).fetchall()
    print(len(result))


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

    perform_3(db)

    # Close the database connection cleanly
    db.close_connection()


if __name__ == '__main__':
    main()
