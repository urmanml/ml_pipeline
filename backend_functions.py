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


def classification_metrics(Y, Y_PRED_prob, threshold= .5, pos_label= 1, cost_fp_fn=(5, 1)):
    import pandas as pd
    from sklearn.metrics.classification import confusion_matrix, f1_score, \
        accuracy_score, balanced_accuracy_score, classification_report, recall_score, \
        precision_score, precision_recall_fscore_support
    from sklearn.metrics import precision_recall_curve, roc_auc_score, accuracy_score
    import numpy as np
    Y_PRED= np.where(Y_PRED_prob> threshold, 1, 0).astype('int')
    Y= Y.astype('int')
    metrics=pd.Series(dtype= object)
    metrics['accuracy_score']= accuracy_score(Y,Y_PRED)
    metrics['f1_score']= f1_score(Y,Y_PRED)
    #metrics['precision_recall_fscore_support']= precision_recall_fscore_support(Y,Y_PRED)
    metrics['recall_score']= recall_score(Y,Y_PRED, pos_label= pos_label)
    metrics['precision_score']= precision_score(Y,Y_PRED, pos_label= pos_label)
    metrics['auc']= roc_auc_score(Y,Y_PRED)
    ###custom metric loss_index

    ###precision recall curve
    precisions, recalls, thresholds= precision_recall_curve(Y, Y_PRED_prob)

    #metrics['prec_recall_plot'].show()
    metrics['gini'] = 2 * roc_auc_score(Y, Y_PRED_prob) - 1
    metrics['confusion_matrix']= confusion_matrix(Y, Y_PRED,labels=[0,1])

    return metrics

def evaluate_estimators(test_X, test_Y, estimator_series, cost_fp_fn=(5,1)):
    """
    Y_PRED_prob= estimator.predict_proba(test_X)[:,1]
    metrics= functions.classification_metrics(test_Y, Y_PRED_prob, threshold=.50,  cost_fp_fn=(5,1))
    """
    import pandas as pd
    Y_PRED_prob_list= list(map(lambda x: x.predict_proba(test_X)[:,1], estimator_series))
    func= lambda x: classification_metrics(test_Y, x, threshold=.50,  cost_fp_fn= cost_fp_fn)
    metrics_list= list(map(func, Y_PRED_prob_list))
    metrics_series= pd.Series(metrics_list)
    metrics_series= metrics_series.set_axis(estimator_series.keys())

    return metrics_series



def define_param_grid(estimator):
    if estimator.__class__.__name__ == 'RandomForestClassifier':
        param_grid = {
            'max_depth': [2, 5, 10, 20],
            'n_estimators': [10, 100, 500]
        }
    elif estimator.__class__.__name__ == 'LogisticRegression':
        param_grid = {
            'penalty': ['l1','l2']
        }
    elif estimator.__class__.__name__ == 'DecisionTreeClassifier':
        param_grid = {
            'max_depth':[ 5, 10, 20, 50],
            'min_samples_leaf': [1, 5, 10]
        }
    elif estimator.__class__.__name__ == 'XGBClassifier':
        param_grid = {"learning_rate": [0.1, 0.01, 0.001],
                  "gamma": [0.01, 0.1, 0.3, 0.5, 1, 1.5, 2],
                  "max_depth": [2, 4, 7, 10],
                  "colsample_bytree": [0.3, 0.6, 0.8, 1.0],
                  "subsample": [0.2, 0.4, 0.5, 0.6, 0.7],
                  "reg_alpha": [0, 0.5, 1],
                  "reg_lambda": [1, 1.5, 2, 3, 4.5],
                  "min_child_weight": [1, 3, 5, 7],
                  "n_estimators": [100, 250, 500, 1000]}
    else: print("hyperparameters not found")
    return param_grid


#define_param_grid(estimator)
def tune_estimator(X, Y, estimator, n_iter= 5, n_jobs= -1):
        #function for a sigle estimator
        #estimator= selected_estimator
        from sklearn.model_selection import RandomizedSearchCV
        param_grid = define_param_grid(estimator)
        estimator = RandomizedSearchCV(estimator, param_distributions= param_grid,cv=5, verbose=1, random_state=1234, n_iter=n_iter, scoring=['accuracy', 'precision'], refit='accuracy', n_jobs= n_jobs)
        estimator.fit(X,Y)
        estimator= estimator.best_estimator_
        return estimator



def tune_estimators(X, Y, estimator_series, n_iter= 5, n_jobs= -1):
    """hyperparameter tuning """

    func= lambda estimator: tune_estimator(X, Y, estimator, n_iter, n_jobs)
    estimator_list= list(map(func, estimator_series))
    estimator_series.set_axis(estimator_series.keys())
    return estimator_series

#
#
# with open(path, 'w') as fp:
#     # To write data to new file uncomment
#     fp.write("New file created")
#     import sys
#     sys.stdout= open(path, 'w')
#     sys.stdout.close()
#     fp.close()
#     sys.stdout.close()