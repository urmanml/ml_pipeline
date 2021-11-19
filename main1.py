import uvicorn
import fastapi
from typing import List, Optional
from fastapi import FastAPI, Query
#from pydantic import BaseModel
## change to project directory here
project_dir= "E:\Python WD/ml_pipeline"
import os
os.chdir(project_dir)

import local_functions
import importlib, joblib
import numpy as np, pandas as pd
importlib.reload(local_functions)
from twitter_sentiment import sentiment_analysis



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
    result= local_functions.adder(num1, num2)
    return result


@app.get("/read_data")
async def read_data_api(path= 'input/dataset.csv'):
    """
    read dataset api
    :param path: path where test dataset is stored
    :return:
    """
    df= local_functions.read_data(path)
    joblib.dump(df,'temporary_objects/df')
    return "data read successfully. Dataframe size is: "+str(df.shape)+" with the following columns: "+str(df.columns.to_list())


async def read_data_koalas_api(path= 'input/dataset.csv'):
    df= local_functions.read_data_koalas(path)
    joblib.dump(df,'temporary_objects/df')
    return "data read successfully. Dataframe size is: "+str(df.shape)+" with the following columns: "+str(df.columns.to_list())


@app.get("/train_test_split")
async def train_test_split_api(target='Outcome', test_ratio= .3):
    """
    train test data split
    :param target: target variable name
    :param test_ratio: test ratio to be maintained
    :return:
    """
    dataset = joblib.load('temporary_objects/df')
    series, message= local_functions.split_data_X_Y(dataset, target, test_size= .3)
    joblib.dump(series, 'temporary_objects/XY')
    return message

@app.get("/define_and_fit_estimators")
async def define_and_fit_estimators_api(Logistic_Regression: bool= True, Decision_tree: bool = True, Xgboost:bool = True):
    """
    Select which estimators need to be considered
    :param Logistic_Regression: LR
    :param Decision_tree: DT
    :param Xgboost: XGB
    :return:
    """
    estimator_flag_list=[Logistic_Regression, Decision_tree, Xgboost]
    estimator_list= local_functions.define_estimators(estimator_flag_list)
    joblib.dump(estimator_list, 'temporary_objects/estimator_list')
    series =joblib.load('temporary_objects/XY')

    estimator_list = joblib.load('temporary_objects/estimator_list')

    X, Y= series['X'], series['Y']
    estimator_list= local_functions.fit_estimators(X, Y, estimator_list=estimator_list)
    joblib.dump(estimator_list, 'temporary_objects/estimator_list')
    return "successfully defined and fitted "+str(len(estimator_list))+" estimators"


async def fit_estimators_api():
    series =joblib.load('temporary_objects/XY')
    estimator_list = joblib.load('temporary_objects/estimator_list')

    X, Y= series['X'], series['Y']
    estimator_list= local_functions.fit_estimators(X, Y, estimator_list=estimator_list)
    joblib.dump(estimator_list, 'temporary_objects/estimator_list')
    return "estimators fitted successfully"

@app.get("/evaluate_estimators")
async def evaluate_estimators_api():
    """
    Evaluate the performance of the fitted estimators
    :return:
    """
    series =joblib.load('temporary_objects/XY')
    estimator_list = joblib.load('temporary_objects/estimator_list')

    test_X, test_Y= series['test_X'], series['test_Y']
    metrics_list= local_functions.evaluate_estimators(test_X, test_Y, estimator_list=estimator_list)
    import pandas as pd

    metrics_df= pd.DataFrame(index= metrics_list[0].index)
    metrics_df.index.name= "Metric"
    metrics_df['Logistic Regression'] = metrics_list[2]
    metrics_df['Decision Tree'] = metrics_list[2]
    metrics_df['XGBoost']= metrics_list[2]
    metrics_df.to_csv('output/metric_df.csv')

    return "test f1 score is: \n xgboost accuracy_score: "+str(metrics_list[2]['accuracy_score'])+"\n Decision Tree accuracy_score: "+str(metrics_list[1]['accuracy_score'])+"\n Logistic Regression accuracy_score: "+str(metrics_list[0]['accuracy_score'])

@app.get("/select_estimator")
async def select_estimator_api(selected_estimator_id: int = 2):
    """
    Select out of the available estimators
    :param selected_estimator_id:
    :return:
    """
    estimator_list = joblib.load('temporary_objects/estimator_list')
    selected_estimator= [estimator_list[selected_estimator_id]]
    joblib.dump(selected_estimator, 'temporary_objects/selected_estimator')
    return "estimator selected: "+ str(selected_estimator[0].__class__.__name__)

@app.get("/tune_estimator")
async def tune_estimator_api(n_iter: int= 10):

    selected_estimator = joblib.load('temporary_objects/selected_estimator')
    series =joblib.load('temporary_objects/XY')
    X, Y, test_X, test_Y= series['X'], series['Y'], series['test_X'], series['test_Y']
    tuned_estimator= local_functions.tune_estimators(X, Y, selected_estimator, n_iter= n_iter)
    joblib.dump(tuned_estimator, 'temporary_objects/tuned_estimator')
    metrics_list= local_functions.evaluate_estimators(test_X, test_Y, estimator_list= tuned_estimator)
    joblib.dump(metrics_list, 'temporary_objects/metrics_list' )

    return "Following hypermarameters are tuned: "+ str(tuned_estimator[0].get_params()) +"\t"+" performance of the model is: \t"+"accuracy_score: "+str(metrics_list[0]['accuracy_score'])





@app.get("/read_prediction_data")
async def read_prediction_data_api(path= 'input/pred_dataset.csv'):
    pred_X= local_functions.read_data(path)
    joblib.dump(pred_X, 'temporary_objects/pred_X')
    return "data read successfully. Dataframe size is: "+str(pred_X.shape)+" with the following columns: "+str(pred_X.columns.to_list())

@app.get("/predict_proba")
async def predict_proba_api(threshold: float= .5):
    tuned_estimator = joblib.load('temporary_objects/tuned_estimator')
    pred_X =joblib.load('temporary_objects/pred_X')
    pred_Y_prob= local_functions.predict_proba(pred_X, tuned_estimator[0])

    pred_X['pred_prob']= pred_Y_prob
    pred_Y = np.where(pred_Y_prob > threshold, 1, 0).astype('int')
    pred_X['pred_class'] = pred_Y
    pred_X.to_csv('output/perdicted_set.csv', index= False)
    pred_X[['pred_prob','pred_class']].to_csv('output/predicted_output.csv')
    return "predicted successfully for: "+ str(pred_X.shape[0])+ " rows"

@app.get("/twitter_sentiment_prediction")
async def twitter_sentiment_api(tweet: str= '#sampletweet'):
    df= pd.DataFrame(columns=['tweet'])
    df.loc[0,'tweet']= tweet
    result= sentiment_analysis(df).sentiment[0]
    return result

@app.get("/sentiment_prediction")
async def twitter_sentiment_api(tweet: str= '#sampletweet'):
    df= pd.DataFrame(columns=['tweet'])
    df.loc[0,'tweet']= tweet
    polarity= sentiment_analysis(df).TextBlob_Polarity
    subjectivity = sentiment_analysis(df).TextBlob_Subjectivity
    return "polarity: "+str(polarity[0])+" and subjectivity: "+str(subjectivity[0])




#from invalid_data_type import dtype_mismatch2


#
# desired_datatype_dict= {'sepal_length': 'float',
#  'sepal_width': 'float',
#  'petal_length': 'float',
#  'petal_width': 'float',
#  'species': 'str',
#  'col_date':'datetime'}
#path= 'E://Python WD/ml_pipeline/input/sample_data.csv'

# @app.get("/dtype_mismatch")
# async def dtype_mismatch_api(data_path: str= 'E://Python WD/ml_pipeline/input/sample_data.csv', desired_datatype_dict= desired_datatype_dict):
#     data= pd.read_csv(data_path, index_col=None)
#     data= data.drop(columns=['Unnamed: 0'])
#     result_dict= dtype_mismatch2(data, desired_datatype_dict)
#     return result_dict
#


#### wrangling api

from wrangling import duplicate_column, IQR_graph_data, find_outliers


@app.get("/duplicate_column")
async def duplicate_column_api(col, data_path: str= 'E://Python WD/ml_pipeline/input/sample_data.csv', output_path: str= 'E://Python WD/ml_pipeline/output/sample_output.csv'):
    data= pd.read_csv(data_path, index_col=None)
    data= duplicate_column(data, col, suffix='_dup')
    data.to_csv(output_path)
    return "success"


@app.get("/IQR_graph_data")
def IQR_graph_data_api(col, data_path: str= 'E://Python WD/ml_pipeline/input/sample_data.csv'):
    data= pd.read_csv(data_path, index_col=None)
    result_dict= IQR_graph_data(data, col)
    import json
    return json.dumps(result_dict)

@app.get("/find_outliers")
def find_outliers_api(col, data_path: str= 'E://Python WD/ml_pipeline/input/sample_data.csv'):
    data= pd.read_csv(data_path, index_col=None)
    result_dict= find_outliers(data, col)
    #create json from dict
    import json
    return json.dumps(result_dict)






