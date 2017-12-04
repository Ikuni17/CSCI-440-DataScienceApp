
/*
Question 1 is interested in the mean revenue, per genre. Since the genre table is normalized this query
requires post processing to get the data in a format which can used for a One-Way ANOVA model.
 */
SELECT DISTINCT K.Tconst, K.Revenue, G.Genre
FROM KAGGLE K, Genre G
WHERE K.Tconst = G.Tconst
AND K.Revenue IS NOT NULL;

/*
Question 2 is interested in the ability of a LogisticRegression model to predict a title's type. The query retrieves
each titles attributes contained in the IMDB database (no Kaggle data), and ensures that no NULL values are returned
in order to ensure the model has complete data for predictions.
 */
SELECT DISTINCT Title_type, Is_adult, R.Avg_rating, Start_year, Runtime
FROM IMDB I, RATINGS R
WHERE I.Tconst = R.Tconst
AND Start_year IS NOT NULL
AND Runtime IS NOT NULL
AND Avg_rating IS NOT NULL
AND Title_type IS NOT NULL
AND Is_adult IS NOT NULL;

/*
Question 3 is determining if there is a linear relationship between the number of seasons and the average rating
of a TV show. This is used within a linear regression model. Season numbers are limited to 50 because there are
a few erroneous rows with 1000 seasons.
 */
SELECT DISTINCT E.Season_Num, R.Avg_rating
FROM EPISODE E, RATINGS R
WHERE E.Econst = R.Tconst
AND E.Season_Num IS NOT NULL
AND E.Season_Num < 50
AND R.Avg_rating IS NOT NULL;

/*
Question 4 is determining the features with the most influence on revenue through FeatureSelection. These features are
then used to build a neural network and predict revenue. All features are ensured to not be NULL in order to ensure
all data entries are complete, and we can accurately compare the effects each feature has on the data.
 */
SELECT DISTINCT Revenue, Start_year, Runtime, Face_number, FB_likes, Rank, Meta_score
FROM IMDB I, KAGGLE K
WHERE I.Tconst = K.Tconst
AND Revenue IS NOT NULL
AND Start_year IS NOT NULL
AND Runtime IS NOT NULL
AND Face_number IS NOT NULL
AND Budget IS NOT NULL
AND FB_likes IS NOT NULL
AND Rank IS NOT NULL
AND Meta_score IS NOT NULL;

/*
Question 5 is a different approach to predicting revenue. The variables are used within a Multiple Linear
Regression model, with content-rating as an indicator variable.
 */
SELECT DISTINCT K.Revenue, K.Budget, K.Content_rating, R.Avg_rating
FROM KAGGLE K, RATINGS R
WHERE K.Tconst = R.Tconst
AND K.Revenue IS NOT NULL
AND K.Budget IS NOT NULL
AND K.Content_rating IS NOT NULL;

