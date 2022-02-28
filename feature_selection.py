
# import joblib
# series = joblib.load('temporary_objects/XY')
# X, Y, test_X, test_Y = series['X'], series['Y'], series['test_X'], series['test_Y']


def feature_suggestion(X, Y, approach= 'multivariate'):
    if approach=='univariate':
        result= univariate_selector(X, Y)
    elif approach=='multivariate':
        result= multivariate_selector(X, Y)
    return result

#feature_suggestion(X, Y)


def univariate_selector(X, Y):
    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import chi2
    import pandas as pd
    # apply SelectKBest class to extract top 10 best features
    bestfeatures = SelectKBest(k='all', score_func=chi2)
    fit = bestfeatures.fit(X, Y)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(X.columns)
    # concat two dataframes for better visualization
    featureScores = pd.concat([dfcolumns, dfscores], axis=1)
    featureScores.columns = ['Feature', 'Score']  # naming the dataframe columns
    result= featureScores.sort_values(by='Score', ascending=False)
    result['Rank']= result['Score'].rank(ascending=False).drop(columns='Score')
    return result[['Feature', 'Rank']]


def multivariate_selector(X, Y):
    from sklearn.linear_model import LogisticRegression
    from sklearn.feature_selection import SelectFromModel
    from sklearn.feature_selection import RFE

    # #Selecting the Best important features according to Logistic Regression using SelectFromModel
    selector = RFE(estimator= LogisticRegression(),n_features_to_select=1)

    selector.fit(X, Y)
    import pandas as pd
    result= pd.DataFrame({'Feature':X.columns, 'Rank': selector.ranking_}).sort_values(by= 'Rank')
    return result

