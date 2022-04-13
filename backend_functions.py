def adder(num1: float= 2, num2: float= 3):
    print('hello')
    return num1+num2

import joblib
def split_data_X_Y(dataset, target, selected_columns ,test_size= .3,  seed= 1234  ):
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
    #etrics['confusion_matrix']= confusion_matrix(Y, Y_PRED,labels=[0,1])
    positives, negatives = confusion_matrix(Y, Y_PRED,labels=[0,1])
    TP, FP= positives[0],positives[1]
    FN, TN= negatives[0], negatives[1]

    metrics['TP']= TP
    metrics['FP']= FP
    metrics['FN']= FN
    metrics['TN']= TN

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



def define_param_grid(model_name):
    from scipy import stats
    dict_hyparameters = {
        'RandomForestClassifier': {
            'max_depth': stats.randint(2, 40),
            'n_estimators': stats.randint(10, 1000),
            'min_samples_split': stats.uniform(0.0001, 0.5),
            'min_samples_leaf': stats.uniform(0.0001, 0.3),
            'max_features': ['auto', 'sqrt', 'log2', None],
            'class_weight': ['balanced', 'balanced_subsample', None]
        },
        'LogisticRegression': {
            'C': stats.uniform(0.001, 5),
            'fit_intercept': [True, False],
            'penalty': ['l1', 'l2', 'elasticnet', 'none'],
            'solver': ['saga'],
            'class_weight': ['balanced', None]
        },
        'DecisionTreeClassifier': {  # 'DecisionTree': { #
            'max_depth': stats.randint(2, 40),
            'min_samples_split': stats.uniform(0.0001, 0.5),
            'min_samples_leaf': stats.uniform(0.0001, 0.3),
            'max_features': ['auto', 'sqrt', 'log2', None],
            'class_weight': ['balanced', None]
        },
        'XGBClassifier': {"learning_rate": stats.uniform(0, 1),  # XGBClassifier
                          "gamma": stats.uniform(0, 10),
                          "max_depth": stats.randint(2, 40),
                          "colsample_bytree": stats.uniform(0, 0.99999),
                          "subsample": stats.uniform(0, 0.99999),
                          "reg_alpha": stats.uniform(0, 1),
                          "reg_lambda": stats.uniform(0.01, 10),
                          "min_child_weight": stats.uniform(0.01, 10),
                          "n_estimators": stats.randint(10, 1000)
                          }
    }

    if model_name not in ['RandomForestClassifier', 'LogisticRegression', 'DecisionTreeClassifier',
                          'XGBClassifier']:  # 'DecisionTree', 'XGBoost']:#,
        print("hyperparameters not found")
        raise NameError('MODEL NAME')

    param_grid = dict_hyparameters[model_name]

    return param_grid


#define_param_grid(estimator)
# log_file_path=
def tune_estimator(X, Y, estimator, log_file_path, n_iter= 5, n_jobs= -1):
        #function for a sigle estimator
        #estimator= selected_estimator
        from sklearn.model_selection import RandomizedSearchCV
        model_name = estimator.__class__.__name__
        param_grid = define_param_grid(estimator)
        estimator = RandomizedSearchCV(estimator, param_distributions= param_grid,cv=5, verbose=50, random_state=1234, n_iter=n_iter, scoring=['accuracy', 'precision'], refit='accuracy', n_jobs= n_jobs)
        import sys
        sys.stdout= open(log_file_path,'w')
        estimator.fit(X,Y)
        sys.stdout.close()
        best_params= estimator.best_params_
        estimator= estimator.best_estimator_

        return estimator, best_params


def tune_estimators(X, Y, estimator_series,log_file_path, n_iter= 5, n_jobs= -1):
    """hyperparameter tuning """
    import pandas as pd
    func= lambda estimator: tune_estimator(X, Y, estimator, log_file_path,n_iter, n_jobs)
    result_list= list(map(func, estimator_series))
    estimator_list=[]
    best_param_list=[]
    for result in result_list:
        estimator_list.append(result[0])
        best_param_list.append(result[1])

    estimator_series= pd.Series(estimator_list)
    best_param_series= pd.Series(best_param_list)
    estimator_series.set_axis(estimator_series.keys())
    best_param_series.set_axis(estimator_series.keys())

    return estimator_series, best_param_series

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

# series =joblib.load('temporary_objects/XY')
# X, Y, test_X, test_Y= series['X'], series['Y'], series['test_X'], series['test_Y']

def recommend_tune_iter(estimator, X):
    import numpy as np
    def compute_searchspace_size_factor(estimator):
        model_name= estimator.__class__.__name__
        param_grid = define_param_grid(model_name)
        n_hyperparameters = len(param_grid.items())
        n_hyperparameters_factor= max(n_hyperparameters-4, 1)
        return n_hyperparameters_factor


    def compute_datset_size_factor(X):
        nrows= X.shape[0]
        ncols= X.shape[1]
        import numpy as np
        dataset_size= nrows* ncols
        datset_size_factor= np.log10(dataset_size)
        return datset_size_factor

    def compute_distribution_factor(estimator):
        total_information = 0
        model_name= estimator.__class__.__name__
        param_grid = define_param_grid(model_name)
        import scipy
        for param in param_grid.keys():
            if param_grid[param].__class__== scipy.stats._distn_infrastructure.rv_frozen:
                information = param_grid[param].entropy().tolist()
            else:
                information= len(param_grid[param])
            total_information += abs(information)

        total_information_factor= total_information
        return total_information_factor


    def compute_iter_new(estimator, X):
        datset_size_factor= compute_datset_size_factor(X)
        distribution_factor= compute_distribution_factor(estimator)
        searchspace_size_factor= compute_searchspace_size_factor(estimator)
        b1, b2, b3= 1/3,1/3,1/3
        net_factor= b1* datset_size_factor+b2*distribution_factor+b3*searchspace_size_factor
        n_iter= round(net_factor* 50,0)
        return n_iter

    n_iter= compute_iter_new(estimator, X)
    return n_iter


#recommend_tune_iter(estimator_series[2], X)

def predict_proba(pred_X, tuned_estimator):
    """
    :param test_X:
    :param estimator_list:
    :return:
    """
    pred_Y_prob= tuned_estimator.predict_proba(pred_X)[:,1]
    return pred_Y_prob