

def split_data_X_Y(dataset, target, test_size= .2,  seed= 1234  ):
    """

    :param dataset: dataset
    :param target: target name
    :param test_size: proportion of test set
    :param seed:
    :return: train test sets
    """
    import pandas as pd
    test_size= test_size
    import random
    from sklearn.model_selection import train_test_split
    # subset only selected columns

    #dataset= dataset[selected_columns+[target]]
    X, test_X,Y,  test_Y = train_test_split(dataset.drop([target], axis= 1), dataset[[target]], test_size=test_size, random_state=seed, stratify= dataset[[target]])
    series= pd.Series(dtype= 'object')
    series['X']=X
    series['Y'] = Y
    series['test_X'] = test_X
    series['test_Y'] = test_Y


    return X, Y, test_X, test_Y

# est_dict= {'CBLOF':True, 'IsolationForest':False, 'COPOD': True, 'LOF':True}

def create_list_of_base_estimators(est_dict, contamination):

    """
    accepts a dict of estimators as being True or False and create a list of base estimators

    :param est_dict:  dict of estimators and corresponding flags
    :param contamination: contamination factor
    :return:
    """
    from pyod.models import suod
    from pyod.models.lof import LOF
    from pyod.models.iforest import IsolationForest
    from pyod.models.cblof import CBLOF
    from pyod.models.copod import COPOD
    base_estimators= []
    for est in est_dict.keys():
        if est_dict[est]== True:
            estimator= eval(est)(contamination=contamination)
            base_estimators.append(estimator)
    return base_estimators


def define_fit_and_approximate_model(base_estimators, contamination, X):
    """

    :param base_estimators: list of base estimators
    :param contamination: contamination factor
    :param X: dataset to be used for training
    :return: fitted model
    """
    from suod.models.base import SUOD

    model = SUOD(base_estimators=base_estimators, n_jobs=len(base_estimators),  # number of workers
                 rp_flag_global=True,  # global flag for random projection
                 bps_flag=False,  # global flag for balanced parallel scheduling
                 approx_flag_global=False,  # global flag for model approximation
                 contamination=contamination)


    model.fit(X)  # fit all models with X
    model.approximate(X)  # conduct model approximation if it is enabled
    return model