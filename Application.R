require(DBI)
require(psych)
require(effects)
require(car)
options(device="png")

imdb <- dbConnect(RSQLite::SQLite(), 'C:\\IMDB\\D3 Python Script\\imdb.db')

db_query <- function(question) {
    if (question == 1) {
        query <- ''
    }else if (question == 2) {
        query <- ''
    }else if (question == 3) {
        query <- ''
    }else if (question == 4) {
        query <- ''
    }else if (question == 5) {
        query <- 'SELECT DISTINCT K.Revenue, K.Budget, K.Content_rating, R.Avg_rating FROM KAGGLE K, RATINGS R WHERE K.Tconst = R.Tconst AND K.Revenue IS NOT NULL AND K.Budget IS NOT NULL AND K.Content_rating IS NOT NULL'
    }

    return(query)
}

perform_5 <- function(){
    query <- db_query(5)
    results <- dbGetQuery(imdb, query)
    results$Content_rating <- as.factor(results$Content_rating)
    lm1 = lm(Revenue ~ Budget + Content_rating + Avg_rating, data = results)
    print(summary(lm1))
    par(mfrow = c(2, 2))
    plot(lm1)
    #plot(allEffects(lm1))
    pairs.panels(results, ellipse = F, main = "Scatterplot matrix")
    print(confint(lm1))
    scatterplot(Revenue~Budget|Content_rating, xlab = "Budget", ylab = "Revenue in Millions", data = results, smooth=F, lwd=3, main="Plot of Budget vs Revenue grouped by Content Rating")
}

perform_5()
dbDisconnect(imdb)
