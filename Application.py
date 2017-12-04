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

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import mutual_info_regression


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
    elif question_num == 6:
        query = 'SELECT DISTINCT Title_type, Is_adult, R.Avg_rating, Start_year, Runtime ' \
                'FROM IMDB I, RATINGS R ' \
                'WHERE I.Tconst = R.Tconst ' \
                'AND Start_year IS NOT NULL ' \
                'AND Runtime IS NOT NULL ' \
                'AND Avg_rating IS NOT NULL ' \
                'AND Title_type IS NOT NULL ' \
                'AND Is_adult IS NOT NULL'


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
    ''' The method analyzes the relationship between Number of Seasons and Show Rating using logistic regression.
        Logistic regression is used to create a line over the data points, which is then visualized through
        plotting the line over the data points, saving the figure to Results3.png '''

    result = query_db(db, 3)
    # Convert to numpy array
    temp_vector = np.fromiter(result.fetchall(), 'i4,f')

    # Split into two vectors
    num_seasons = temp_vector['f0']
    rating = temp_vector['f1']

    plt.figure(figsize=(14, 7))

    # retrieve the attributes of line fit with linear regression
    slope, intercept, r, p, std_error = scipy.stats.linregress(rating, num_seasons)
    # plot the data on the chart
    plt.scatter(rating, num_seasons, label='Data')
    # plot the line on the chart
    plt.plot(rating, intercept + slope * rating, 'r', label="Line: y = {:.3}x + {:.3}\nr = {:.5}\np-value = {:.5}".format(slope, intercept, r, p))

    #print("Slope: {0}\nIntercept: {1}\nr: {2}\np-value: {3}\nstd. error: {4}\n".format(slope, intercept, r, p, std_error))
    plt.legend()
    plt.title('Linear Regression for Number of Seasons vs. Show Rating')
    plt.ylabel('Number of Seasons')
    plt.xlabel('Show Rating')

    plt.savefig('Results3.pdf')
    #plt.show()


def determine_components(data, output, labels, f_regress = False):
    ''' Determine the order that attributes are removed through SelectKBest methods'''

    # order of removal using mutual_info_regression (Estimate mutual information for a continuous target variable)
    initial_values = {x: y for x, y in zip(labels, data[0])}
    ordered_values = []

    # for each possible number of remaining attributes
    for i in reversed(range(len(data[0]))):
        # select the k-best attributes for the data
        new_data = SelectKBest(mutual_info_regression, k=i).fit_transform(data, output)
        # for each remaining attribute
        for key, value in initial_values.items():
            # determine which attribute was not included in the new data, add it to the list
            if value not in new_data[0]:
                ordered_values.insert(0, key)
                del initial_values[key]
                break
    #print(ordered_values)

    if f_regress:
        # order of removal using f_regression (Univariate linear regression tests)
        initial_values = {x: y for x, y in zip(labels, data[0])}
        ordered_values = []

        for i in reversed(range(len(data[0]))):
            new_data = SelectKBest(f_regression, k=i).fit_transform(data, output)
            for key, value in initial_values.items():
                if value not in new_data[0]:
                    ordered_values.insert(0, key)
                    del initial_values[key]
                    break
        #print(ordered_values)
        return ordered_values

    return ordered_values


# Perform analysis specific to question 4: Predict Revenue
def perform_4(db):
    ''' This method analyzes how well a Neural Network can predict the the Revenue of a production using a varying
        number of attributes, which are selected through feature selection. A network is initialized and trained on
        the selected components, then tested, which is quantified with mean squared error. Each network is tested
        using k-fold cross validation, with the avg testing error recorded over the k tests. This testing error is
        then averaged through the number of tests run, then plotted and saved to the file Results4.png.

        Note: extra plt commands were used to tune the network. They are included to show how the network was tuned,
        however they are not being utilized in the final application. '''

    # retrieve the data relevant to this question from the DataBase
    result = query_db(db, 4).fetchall()
    labels = ['Revenue', 'Start_year', 'Runtime', 'Face_number', 'FB_likes', 'Rank', 'Meta_score']

    # f_regression is an alternate feature selection method, however tuning revealed mutual_info_regression to
    # perform better on the selected data. Thus, f_regression is not used in the final application
    f_regress = False

    # split into predicted value revenue and input variables data
    revenue = np.array([x[0] for x in result])
    original_data = [x[1:] for x in result]

    # get the order that attributes will be removed from the
    ordered_labels = determine_components(original_data, revenue, labels[1:], f_regress)

    # intialize the plot and the sum of the testing error
    plt.figure(figsize=(14, 7))
    plt.grid(True)
    axes = plt.gca()
    #axes.set_xlim([0, 7])
    #axes.set_ylim([0, 250000])
    number_of_tests = 100
    trend = {'x': [1,2,3,4,5,6], 'y': [0,0,0,0,0,0]}

    # colors = {0:'k', 1:'b', 2:'g', 3:'r', 4:'c', 5:'y', 6:'m'}

    # run an entire test (training/testing a network on each set of attributes) this many times
    for j in range(number_of_tests):
        output = {'x': [], 'y': []}

        # for each set of principle compenents
        for i in reversed(range(1, 7)):

            #data = [(x[1:]) for x in result]
            #pca = PCA(n_components=i)
            #pca.fit(data)
            #data = pca.transform(data)
            kf = KFold(n_splits=10)
            pca_output = []

            if f_regress:
                data = SelectKBest(f_regression, k=i).fit_transform(original_data, revenue)
            else:
                data = SelectKBest(mutual_info_regression, k=i).fit_transform(original_data, revenue)
            #print(data)


            # run a k-fold cross validation test, tracking mean squared error, on a neural net
            for train_index, test_index in kf.split(data):
                # train the network
                clf = MLPRegressor(alpha=0.01, hidden_layer_sizes=(100,), max_iter=50000, early_stopping=False, batch_size=100,
                                   activation='relu', solver='adam', verbose=False,
                                   learning_rate_init=0.001, learning_rate='adaptive', tol=0.000000000000000000001)
                clf.fit(data[train_index], revenue[train_index])

                # test the network on the test data, calculate mean squared error
                pred_revenue = clf.predict(data[test_index])  # predict network output given input data
                target_revenue = revenue[test_index]
                error = [(pred_revenue[x] - target_revenue[x])**2 for x in range(len(pred_revenue))]
                error = sum(error) / len(error)
                #print("predicted rev = {}".format(pred_revenue))
                #print('target rev = {}'.format(target_revenue))
                #print('error = {}'.format(error))

                pca_output += [error]
                #plt.scatter(i, error, color=colors[j])

            # average the error of the k-folds tests over each component set
            output['y'] += [sum(pca_output) / len(pca_output)]
            output['x'] += [i]
            #print('network outputs = {}'.format(pca_output))
            #print('avg error of folds = {}'.format(sum(pca_output) / len(pca_output)))
            #print(output)


        #trend['y'] = [trend['y'][x] + (output['y'][x] - output['y'][x+1]) for x in range(len(output['y'])-1)]
        # sum the error of each test to be avered and plotted later
        trend['y'] = [trend['y'][x] + output['y'][x] for x in range(len(output['y']))]
        #plt.plot(output['x'], output['y'], color=colors[j])

    #trend['y'] = [(trend['y'][x] / 5) + 150000 for x in range(len(trend['y']))] + [150000]
    # average the testing error of the network trained on each set of components over all the tests run
    trend['y'] = [(trend['y'][x] / number_of_tests) for x in range(len(trend['y']))]
    plt.bar(trend['x'], trend['y'], color='m')

    # label the x-axis with the ordered attributes
    ax = plt.subplot()
    ax.set_xticklabels(['_', ordered_labels[0]] + ['..., ' + ordered_labels[x] for x in range(1, len(ordered_labels))])

    # label and format the plot
    plt.title('Avg Prediction Error of {} Networks Trained on Varying Numbers of Attributes as Selected by Feature Selection'.format(number_of_tests))
    plt.xlabel('Components Included in Test ({})'.format(', '.join(ordered_labels)))
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

def perform_6(db):
    ''' This method analyzes how well Logistic Regression can predict a Title_type of a production.
        The prediction is performed using 4 attributes: Is_adult, R.Avg_rating, Start_year, and Runtime.
        The results of the predictions are recorded in three heat maps, plotted against combination of
        the latter three attributes. These heatmaps show patterns in how logistic regressions was able
        to classify the data in response to the attributes. '''

    result = query_db(db, 6).fetchall()

    type = np.array([x[0] for x in result])
    data = [x[1:] for x in result]

    # create and train the logistic regression module
    model = lm.LogisticRegression()
    model.fit(data, type)
    # print("Model Score is: {}".format(model.score(df, type)))

    # predict the data on model
    predicted_types = model.predict(data)

    # create tables of each attribute
    rating = [x[1] for x in data]
    year = [x[2] for x in data]
    runtime = [x[3] for x in data]
    # create a table representing which classifications the model predicted correctly, used to create a heatmap
    correct = [True if target == predicted else False for target, predicted in zip(type, predicted_types)]
    df = pd.DataFrame({"Rating": rating, "Year": year, "Runtime": runtime, "Percent Correct": correct})


    # create the first plot, plotting each individual's Year against Rating
    plt.figure(figsize=(14, 7))
    plt.title('Heatmap of Correct Predictions Plotted over Rating vs. Year')

    # group the data by (y, x) axis, and the True/False values for the heatmap
    res = df.groupby(['Year', 'Rating'])['Percent Correct'].mean().unstack()
    # manually set the x-ticks for Ratings axis
    xticks = np.arange(10 + 1)
    # plot the heatmap
    ax = seaborn.heatmap(res, linewidth=0, xticklabels=xticks, cbar_kws={'label': 'Percent of Classifications Correct'})
    ax.set_xticks(xticks * ax.get_xlim()[1] / (10))
    ax.invert_yaxis()
    # label the colorbar
    cbar = ax.collections[0].colorbar
    cbar.set_ticks([0, .25, .50, .75, 1])
    cbar.set_ticklabels(['0%', '25%', '50%', '75%', '100%'])

    plt.savefig('Results6-1.png')

    # create the second plot, plotting each individual's Year against Runtime
    plt.figure(figsize=(14, 7))
    plt.title('Heatmap of Correct Predictions Plotted over Runtime vs. Year')

    res = df.groupby(['Year', 'Runtime'])['Percent Correct'].mean().unstack()
    ax = seaborn.heatmap(res, linewidth=0, cbar_kws={'label': 'Percent of Classifications Correct'})
    ax.invert_yaxis()
    cbar = ax.collections[0].colorbar
    cbar.set_ticks([0, .25, .50, .75, 1])
    cbar.set_ticklabels(['0%', '25%', '50%', '75%', '100%'])

    plt.savefig('Results6-2.png')

    # create the third plot, plotting each individual's Runtime against Rating
    plt.figure(figsize=(14, 7))
    plt.title('Heatmap of Correct Predictions Plotted over Rating vs. Runtime')

    res = df.groupby(['Runtime', 'Rating'])['Percent Correct'].mean().unstack()
    xticks = np.arange(10 + 1)
    ax = seaborn.heatmap(res, linewidth=0, xticklabels=xticks, cbar_kws={'label': 'Percent of Classifications Correct'})
    ax.set_xticks(xticks * ax.get_xlim()[1] / (10))
    ax.invert_yaxis()
    cbar = ax.collections[0].colorbar
    cbar.set_ticks([0, .25, .50, .75, 1])
    cbar.set_ticklabels(['0%', '25%', '50%', '75%', '100%'])

    plt.savefig('Results6-3.png')

    return

    encoded_colors = ['k' if target == predicted else 'r' for target, predicted in zip(type, predicted_types)]
    #plt.plot(type, model.predict(data))
    #plt.scatter()
    plt.scatter([x[1] for x in data], [x[2] for x in data], [x[3]/4 for x in data], alpha=0.5, c=encoded_colors)
    # plt.plot(data, model.predict(df), color='r')

    plt.legend()
    plt.ylabel('Year')
    plt.xlabel('Rating')
    plt.savefig('Results6.png')
    plt.show()



    return
    #print(np.unique([x[0] for x in result]))

    type = np.array([x[0] for x in result])
    data = [x[1:] for x in result]

    #df = pd.DataFrame(data)
    model = lm.LogisticRegression()
    model.fit(data, type)



    #plt.scatter([x[0] for x in data], type)
    #plt.plot(data, model.predict(df), color='r')

    plt.figure(figsize=(14, 7))
    #print('mean = {}'.format(stats.mean([int(x) for x in type])))
    #Is_adult, R.Avg_rating, Start_year, Runtime
    encoded_colors = ['k' if x==0 else 'c' for x in type]
    plt.scatter([x[0] for x in data], [x[1] for x in data], s=[x[2]/2 for x in data] ,alpha=0.4, c=encoded_colors)
    #plt.plot(data, model.predict(df), color='r')

    plt.legend()
    plt.ylabel('Rating')
    plt.xlabel('Is Adult')
    plt.savefig('Results6.png')
    plt.show()


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


