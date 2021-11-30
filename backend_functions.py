def adder(num1: float= 2, num2: float= 3):
    print('hello')
    return num1+num2

import joblib
def split_data_X_Y(dataset, target, selected_columns ,test_size= .3,    ):
    import pandas as pd
    test_size= test_size
    import random
    from sklearn.model_selection import train_test_split
    # subset only selected columns

    dataset= dataset[selected_columns+[target]]
    X, test_X,Y,  test_Y = train_test_split(dataset.drop([target], axis= 1), dataset[[target]], test_size=test_size, random_state=seed)
    series= pd.Series(dtype= 'object')
    series['X']=X
    series['Y'] = Y
    series['test_X'] = test_X
    series['test_Y'] = test_Y


    return series, "train test split done successful. Train size: "+str(X.shape)+ ", test size: "+str(test_X.shape)





def define_estimators(estimator_flag_dict, seed= 1234):
    import pandas as pd
    from sklearn.tree import DecisionTreeClassifier
    # from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    import xgboost as xgb
    seed= seed
    estimator_series= pd.Series(dtype= 'object')
    #base models
    if estimator_flag_dict['lr']==True:
        estimator = LogisticRegression(random_state=seed)
        estimator_series['lr']= estimator

    if estimator_flag_dict['dt']==True:
        estimator = DecisionTreeClassifier(random_state=seed)
        estimator_series['dt']= estimator

    if estimator_flag_dict['xgb'] == True:
        estimator = xgb.XGBClassifier(random_state=seed, learning_rate=0.01)
        estimator_series['xgb']= estimator

    #estimator_series = pd.Series(dtype=object)

    #estimator_list= [estimator1, estimator2, estimator3]
    return estimator_series


def fit_estimators(X, Y, estimator_series):
    import pandas as pd
    ### series to list
    fit_func = lambda x: x.fit(X,Y)
    fitted_list= list(map(fit_func, estimator_series))
    result_series= pd.Series(fitted_list)
    result_series= result_series.set_axis(estimator_series.keys())
    estimator_series= result_series
    return estimator_series

