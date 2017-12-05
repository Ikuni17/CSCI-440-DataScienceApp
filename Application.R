# Sam Congdon and Bradley White
# CSCI 440: Data Science Application
# December 3, 2017

require(DBI)
require(psych)
require(car)
options(device="pdf")

# Create the database connection
imdb <- dbConnect(RSQLite::SQLite(), 'C:\\IMDB\\D3 Python Script\\imdb.db')

# Returns the query for a specific question
db_query <- function(question) {
   if (question == 3) {
        query <- 'SELECT DISTINCT E.Season_Num, R.Avg_rating FROM EPISODE E, RATINGS R WHERE E.Econst = R.Tconst AND E.Season_Num IS NOT NULL AND E.Season_Num < 50 AND R.Avg_rating IS NOT NULL'
    }else if (question == 5) {
        query <- 'SELECT DISTINCT K.Revenue, K.Budget, K.Content_rating, R.Avg_rating FROM KAGGLE K, RATINGS R WHERE K.Tconst = R.Tconst AND K.Revenue IS NOT NULL AND K.Budget IS NOT NULL AND K.Content_rating IS NOT NULL'
    }

    return(query)
}

# Get a summary for question 3, used to compliment the analysis within Application.py
perform_3<- function(){
    # Get the correct query and use it to get the results from the database
    query <- db_query(3)
    results <- dbGetQuery(imdb, query)
    # Build a linear model with average rating as the response and season number as the explanatory variable
    lm <- lm(Season_Num ~ Avg_rating, data=results)
    # Print the summmary for the linear model
    print(summary(lm))
    # Print the 95% confidence interval for all variables
    print(confint(lm))
}

# Perform analysis specific to question 5, which is predicting revenue with multiple linear regression based on
# Budget, Content rating, and average user rating
perform_5 <- function(){
    # Get the correct query and use it to get the results from the database
    print("Retrieving question 5 results from database")
    query <- db_query(5)
    results <- dbGetQuery(imdb, query)

    # Convert content rating to an indicator variable
    results$Content_rating <- as.factor(results$Content_rating)
    print("Finished post-processing, building Multiple Linear Regression model")
    # Build the linear model
    lm1 = lm(Revenue ~ Budget + Content_rating + Avg_rating, data = results)
    # Print the summary for the linear model which contains estimated slope and y intercept, std. error, t stats,
    # p values for each explanatory variable, multiple and adjusted R squared and F-stats
    print(summary(lm1))
    # Create the four diagnostic plots in a 2x2 grid and save to file. The four plots are Residuals vs Fitted, Normal
    # Q-Q Plot, Scale Location, and Residuals vs Leverage
    print("Outputting question 5 results to Results directory")
    pdf("Results\\5-Diagnostics.pdf")
    par(mfrow = c(2, 2))
    plot(lm1)
    # Create the pairs panel to look for multicollinearity and see initial r values for all variables
    pdf("Results\\5-PairsPanel.pdf")
    pairs.panels(results, ellipse = F, main = "Scatterplot matrix")
    # Print the 95% confidence interval for all variables
    print(confint(lm1))
    # Create some scatterplot
    pdf("Results\\5-Scatterplot-Budget.pdf")
    scatterplot(Revenue~Budget|Content_rating, xlab = "Budget", ylab = "Revenue in Millions", data = results, smooth=F, lwd=3, main="Plot of Budget vs Revenue grouped by Content Rating")
    pdf("Results\\5-Scatterplot-Rating.pdf")
    scatterplot(Revenue~Avg_rating|Content_rating, xlab = "Average Rating", ylab = "Revenue in Millions", data = results, smooth=F, lwd=3, main="Plot of Average Rating vs Revenue grouped by Content Rating")
}

perform_5()
dbDisconnect(imdb)
