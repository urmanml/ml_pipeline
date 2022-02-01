

def fit_estimators(X, Y, estimator_series, n_iter= 5, n_jobs= -1):
    """hyperparameter tuning """
    import pandas as pd
    func= lambda estimator: fit_estimator(X, Y, estimator, n_iter, n_jobs)
    result_list= list(map(func, estimator_series))
    estimator_list=[]
    for result in result_list:
        estimator_list.append(result)

    estimator_series= pd.Series(estimator_list)
    estimator_series.set_axis(estimator_series.keys())
    return estimator_series

def fit_estimator(X, Y, estimator, n_iter=5, n_jobs=-1):
    # function for a single estimator
    # estimator= selected_estimator
    from sklearn.model_selection import RandomizedSearchCV
    from backend_functions import define_param_grid
    param_grid = define_param_grid(estimator)
    estimator = RandomizedSearchCV(estimator, param_distributions=param_grid, cv=5, verbose=2, random_state=1234,
                                   n_iter=n_iter, scoring=['accuracy'], refit='accuracy',
                                   n_jobs=n_jobs)

    estimator.fit(X, Y)
    estimator = estimator.best_estimator_

    return estimator

#
# import joblib
# series =joblib.load('temporary_objects/XY')
# X, Y = series['X'], series['Y']
# estimator_series = joblib.load('temporary_objects/estimator_series')
# estimator = estimator_series['xgb']


def evaluate_estimator(X, Y, estimator):
    import pandas as pd
    from sklearn.model_selection import cross_validate
    cv_results = cross_validate(estimator, X, Y, cv=3, scoring=['accuracy','precision','recall','f1'], return_train_score=True)
    metrics=pd.Series(dtype= object)
    metrics['accuracy_score']=cv_results['test_accuracy'].mean()
    metrics['precision_score']=cv_results['test_precision'].mean()
    metrics['recall_score']=cv_results['test_recall'].mean()
    metrics['f1_score']=cv_results['test_f1'].mean()
    return metrics
#evaluate_estimator(X, Y, estimator)

def evaluate_estimators(X, Y, estimator_series):
    import pandas as pd
    func= lambda estimator: evaluate_estimator(X, Y, estimator)
    metrics_list= list(map(func, estimator_series))
    metrics_series= pd.Series(metrics_list)
    metrics_series= metrics_series.set_axis(estimator_series.keys())
    return metrics_series
