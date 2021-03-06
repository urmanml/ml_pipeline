import uvicorn
import fastapi
from typing import List, Optional
from fastapi import FastAPI, Query
#from pydantic import BaseModel
## change to project directory here
project_dir= "E:/Python WD/ml_pipeline"
import os
os.chdir(project_dir)

import local_functions
import importlib, joblib
import numpy as np, pandas as pd
import backend_functions
importlib.reload(backend_functions)



app = FastAPI()

@app.get("/print_wd")
async def print_wd_api():
    """
    :return: Returns the cwd directory for debugging and setup purposes
    """
    import os
    return os.getcwd()

@app.get("/adder_api")
async def adder_api(num1: float= 2, num2: float= 3):
    """
    Basic adder function for testing
    :param num1:
    :param num2:
    :return:
    """
    result= backend_functions.adder(num1, num2)
    return result


@app.get("/train_test_split")
async def train_test_split_api(target='Outcome', test_ratio= .3, selected_columns=[], create_validation_set= False):
    """
    train test data split
    :param target: target variable name
    :param test_ratio: test ratio to be maintained
    :return:
    """
    print("websocket message: train test split started")

    dataset = joblib.load('temporary_objects/df')
    selected_columns= list(dataset.columns[0:-1])

    import train_test_split
    series= train_test_split.split_data_X_Y(dataset, target, selected_columns= selected_columns, test_size= .3, create_validation_set= False)

    ### reconstruct train and test set for storing to delta lake
    train_set= pd.concat([series['X'], series['Y']], axis=1)
    test_set= pd.concat([series['test_X'], series['test_Y']], axis=1)
    if create_validation_set== True:
        validation_set = pd.concat([series['val_X'], series['val_Y']], axis=1)

    # Store intermediate objects to specified location
    joblib.dump(series, 'temporary_objects/XY')


    # Store train and test set to delta lake for preview
    ## train_set.to_delta
    ## train_set.to_delta



    ## websocket message here to signal completion
    print("websocket message: train test split started")

    return message



@app.get("/feature_suggestion")
async def feature_suggestion_api():
    series= joblib.load('temporary_objects/XY')
    X, Y = series['X'], series['Y']
    import feature_selection
    result= feature_selection.feature_suggestion(X, Y, approach= 'multivariate')
    import json
    return result.to_json()


@app.get("/recommend_balancing")
async def recommend_balancing_api():
    series= joblib.load('temporary_objects/XY')
    Y = series['Y']
    import recommend_balancing
    result = recommend_balancing.recommend_balancing(Y)
    result
    return result



@app.get("/class_imbalance")
async def class_imbalance_api(proportion= .5):
    series= joblib.load('temporary_objects/XY')
    X, Y = series['X'], series['Y']
    import class_imbalance
    proportion= float(proportion)
    X, Y= class_imbalance.class_imbalance(X, Y, prop= proportion)
    series['X'], series['Y']= X, Y
    joblib.dump(series, 'temporary_objects/XY_balanced')
    return "success"



@app.get("/class_distribution")
async def class_distribution_api():
    series= joblib.load('temporary_objects/XY')
    Y = series['Y']
    import class_distribution
    result = class_distribution.class_distribution(Y)
    result
    import json
    result.to_json()
    return result.to_json()



@app.get("/define_and_fit_estimators")
async def define_and_fit_estimators_api(estimator_list:str= ['lr','xgb', 'dt']):
    """
    Select which estimators need to be considered
    :param Logistic_Regression: LR
    :param Decision_tree: DT
    :param Xgboost: XGB
    :return:
    """

    estimator_flag_dict={'lr':False, 'dt':False, 'xgb':False}

    for estimator in estimator_list:
        estimator_flag_dict[estimator]= True

#    estimator_flag_dict={'lr':True, 'dt':True, 'xgb':True}
    estimator_series= backend_functions.define_estimators(estimator_flag_dict)

    joblib.dump(estimator_series, 'temporary_objects/estimator_series')

    ################ fit estimator
    series =joblib.load('temporary_objects/XY')

    estimator_series = joblib.load('temporary_objects/estimator_series')

    X, Y= series['X'], series['Y']
    import fit_estimators
    estimator_series= fit_estimators.fit_estimators(X, Y, estimator_series=estimator_series)
    joblib.dump(estimator_series, 'temporary_objects/estimator_series')
    return "successfully defined and fitted "+str(len(estimator_series))+" estimators"

#list(s1.keys())
#list(s1.values)

@app.get("/evaluate_estimators")
async def evaluate_estimators_api():
    """
    Evaluate the performance of the fitted estimators
    :return:
    """
    import pandas as pd
    import fit_estimators
    series =joblib.load('temporary_objects/XY')
    estimator_series = joblib.load('temporary_objects/estimator_series')
    test_X, test_Y= series['test_X'], series['test_Y']
    metrics_series= fit_estimators.evaluate_estimators(test_X, test_Y, estimator_series=estimator_series)



    metrics_df= pd.DataFrame(columns=metrics_series.keys(), index=metrics_series[0].index)
    metrics_df.index.name= "Metric"

    for col in metrics_series.keys():
        metrics_df[col]= metrics_series[col]




    metrics_df.to_csv('output/metric_df.csv')




    return "success"


@app.get("/recommend_estimator")
async def recommend_estimator_api():
    """recommend estimator to be selected after they have been evaluated"""
    path= 'output/metric_df.csv'
    import pandas as pd
    metrics_df= pd.read_csv(path)
    import recommend_estimator
    result= recommend_estimator.recommend_estimator(metrics_df)
    return result


@app.get("/tune_estimator")
async def tune_estimator_api(selected_estimator_id='xgb', n_iter: int= 10):

    import joblib
    estimator_series = joblib.load('temporary_objects/estimator_series')
    selected_estimator_series= pd.Series(estimator_series[selected_estimator_id])
    selected_estimator_series= selected_estimator_series.set_axis([selected_estimator_id])

    series =joblib.load('temporary_objects/XY')
    X, Y, test_X, test_Y= series['X'], series['Y'], series['test_X'], series['test_Y']
    log_file_path= 'temporary_objects/tune_log.txt'
    tuned_estimator_series, best_param_series= backend_functions.tune_estimators(X, Y, selected_estimator_series,log_file_path, n_iter= n_iter)
    joblib.dump(tuned_estimator_series, 'temporary_objects/tuned_estimator_series')
    tuned_metrics_series= backend_functions.evaluate_estimators(test_X, test_Y, estimator_series= tuned_estimator_series)
    joblib.dump(tuned_metrics_series, 'temporary_objects/tuned_metrics_series' )
    import json
    return json.dumps(best_param_series[0])

@app.get("/recommend_tune_iter")
async def recommend_tune_iter_api(selected_estimator_id='dt'):

    import joblib, pandas as pd
    estimator_series = joblib.load('temporary_objects/estimator_series')
    selected_estimator_series= pd.Series(estimator_series[selected_estimator_id])
    selected_estimator_series= selected_estimator_series.set_axis([selected_estimator_id])
    estimator= selected_estimator_series[selected_estimator_id]
    series =joblib.load('temporary_objects/XY')
    X, Y, test_X, test_Y= series['X'], series['Y'], series['test_X'], series['test_Y']

    import recommend_tune_iterations
    n_tune_iter= recommend_tune_iterations.recommend_tune_iter(estimator, X)
    return n_tune_iter





@app.get("/read_prediction_data")
async def read_prediction_data_api():
    pred_X= local_functions.read_data(path)
    joblib.dump(pred_X, 'temporary_objects/pred_X')
    return "data read successfully. Dataframe size is: "+str(pred_X.shape)+" with the following columns: "+str(pred_X.columns.to_list())



@app.get("/predict_proba")
async def predict_proba_api(path= 'input/pred_dataset.csv', threshold: float= .5):
    pred_X= pd.read_csv(path)
    tuned_estimator_series = joblib.load('temporary_objects/tuned_estimator_series')

    pred_Y_prob= backend_functions.predict_proba(pred_X, tuned_estimator_series[0])

    pred_X['pred_prob']= pred_Y_prob
    pred_Y = np.where(pred_Y_prob > threshold, 1, 0).astype('int')
    pred_X['pred_class'] = pred_Y
    pred_X.to_csv('output/perdicted_set.csv', index= False)
    pred_X[['pred_prob','pred_class']].to_csv('output/predicted_output.csv')
    return "predicted successfully for: "+ str(pred_X.shape[0])+ " rows"



def correlation_api(path= 'temporary_objects/df', method= 'spearman', columns= ['Pregnancies'  ,'Glucose']):
    dataset = joblib.load(path)

    import correlation
    #methods= ['pearson','kendall','spearman']
    result= correlation.correlation(dataset, method= method, columns= columns)
    return result



















