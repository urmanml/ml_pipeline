

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
    from sklearn.metrics import confusion_matrix, make_scorer, balanced_accuracy_score

    scoring_dict = {'accuracy_score': 'accuracy', 'precision_score': 'precision', 'recall_score': 'recall',
                    'f1_score': 'f1', 'auc': 'roc_auc',
                    'TP': make_scorer(confusion_matrix_TP), 'FP': make_scorer(confusion_matrix_FP),
                    'FN': make_scorer(confusion_matrix_FN), 'TN': make_scorer(confusion_matrix_TN)}

    cv_results = cross_validate(estimator, X, Y, cv=5, scoring=scoring_dict, return_train_score=False)
    metrics=pd.Series(dtype= object)
    metrics['accuracy_score']=cv_results['test_accuracy_score'].mean()
    metrics['precision_score']=cv_results['test_precision_score'].mean()
    metrics['recall_score']=cv_results['test_recall_score'].mean()
    metrics['f1_score']=cv_results['test_f1_score'].mean()
    metrics['roc']= cv_results['test_auc'].mean()
    metrics['TP']= cv_results['test_TP'].sum().astype('int')
    metrics['FP']= cv_results['test_FP'].sum().astype('int')
    metrics['FN']= cv_results['test_FN'].sum().astype('int')
    metrics['TN']= cv_results['test_TN'].sum().astype('int')

    return metrics
#evaluate_estimator(X, Y, estimator)

def evaluate_estimators(X, Y, estimator_series):
    import pandas as pd
    func= lambda estimator: evaluate_estimator(X, Y, estimator)
    metrics_list= list(map(func, estimator_series))
    metrics_series= pd.Series(metrics_list)
    metrics_series= metrics_series.set_axis(estimator_series.keys())
    return metrics_series

#####################################################
def my_custom_loss_func(y_true, y_pred):
    diff = np.abs(y_true.values - y_pred).max()
    return np.log1p(diff)
#my_custom_loss_func()


def confusion_matrix_TP(y_true, y_pred):
    y_true= y_true.values
    from sklearn.metrics import confusion_matrix
    positives, negatives = confusion_matrix(y_true, y_pred)
    TP, FP = positives[0], positives[1]
    FN, TN = negatives[0], negatives[1]

    return TP

def confusion_matrix_FP(y_true, y_pred):
    y_true= y_true.values
    from sklearn.metrics import confusion_matrix
    positives, negatives = confusion_matrix(y_true, y_pred)
    TP, FP = positives[0], positives[1]
    FN, TN = negatives[0], negatives[1]

    return FP

def confusion_matrix_FN(y_true, y_pred):
    y_true= y_true.values
    from sklearn.metrics import confusion_matrix
    positives, negatives = confusion_matrix(y_true, y_pred)
    TP, FP = positives[0], positives[1]
    FN, TN = negatives[0], negatives[1]

    return FN

def confusion_matrix_TN(y_true, y_pred):
    y_true= y_true.values
    from sklearn.metrics import confusion_matrix
    positives, negatives = confusion_matrix(y_true, y_pred)
    TP, FP = positives[0], positives[1]
    FN, TN = negatives[0], negatives[1]

    return TN






