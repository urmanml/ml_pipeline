

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
        ### giving different weights to different factors
        b1, b2, b3= 1/3,1/3,1/3
        net_factor= b1* datset_size_factor+b2*distribution_factor+b3*searchspace_size_factor
        n_iter= round(net_factor* 50,0)
        return n_iter

    n_iter= compute_iter_new(estimator, X)
    return n_iter





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






