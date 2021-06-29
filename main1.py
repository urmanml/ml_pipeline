import uvicorn
import fastapi
from typing import List, Optional
from fastapi import FastAPI, Query
from pydantic import BaseModel
#cwd= "E:\Python WD\LinearRegression"
import os
#os.chdir(cwd)
#os.getcwd()
from ml_pipeline import local_functions
import importlib, joblib
importlib.reload(local_functions)

app = FastAPI()

async def adder_api(num1: float= 2, num2: float= 3):
    result= local_functions.adder(num1, num2)
    return result

@app.get("/read_data")
async def read_data_api(path= 'ml_pipeline/dataset.csv'):
    df= local_functions.read_data(path)
    joblib.dump(df,'ml_pipeline/temporary_objects/df')
    return "data read successfully. Dataframe size is: "+str(df.shape)+" with the following columns: "+str(df.columns.to_list())


async def read_data_koalas_api(path= 'ml_pipeline/dataset.csv'):
    df= local_functions.read_data_koalas(path)
    joblib.dump(df,'ml_pipeline/temporary_objects/df')
    return "data read successfully. Dataframe size is: "+str(df.shape)+" with the following columns: "+str(df.columns.to_list())


@app.get("/train_test_split")
async def train_test_split_api(target='Outcome', test_ratio= .3):
    dataset = joblib.load('ml_pipeline/temporary_objects/df')
    series, message= local_functions.split_data_X_Y(dataset, target, test_size= .3)
    joblib.dump(series, 'ml_pipeline/temporary_objects/XY')
    return message

@app.get("/define_estimators")
async def define_estimators_api(Logistic_Regression: bool= True, Decision_tree: bool = True, Xgboost:bool = True):
    estimator_flag_list=[Logistic_Regression, Decision_tree, Xgboost]
    estimator_list= local_functions.define_estimators(estimator_flag_list)
    joblib.dump(estimator_list, 'ml_pipeline/temporary_objects/estimator_list')
    return "successfully defined "+str(len(estimator_list))+" estimators"


@app.get("/fit_estimators")
async def fit_estimators_api():
    series =joblib.load('ml_pipeline/temporary_objects/XY')
    estimator_list = joblib.load('ml_pipeline/temporary_objects/estimator_list')

    X, Y= series['X'], series['Y']
    estimator_list= local_functions.fit_estimators(X, Y, estimator_list=estimator_list)
    joblib.dump(estimator_list, 'ml_pipeline/temporary_objects/estimator_list')
    return "estimators fitted successfully"

@app.get("/evaluate_estimators")
async def evaluate_estimators_api():
    series =joblib.load('ml_pipeline/temporary_objects/XY')
    estimator_list = joblib.load('ml_pipeline/temporary_objects/estimator_list')

    test_X, test_Y= series['test_X'], series['test_Y']
    metrics_list= local_functions.evaluate_estimators(test_X, test_Y, estimator_list=estimator_list)


    return "test f1 score is: \n xgboost accuracy_score: "+str(metrics_list[2]['accuracy_score'])+"\n Decision Tree accuracy_score: "+str(metrics_list[1]['accuracy_score'])+"\n Logistic Regression accuracy_score: "+str(metrics_list[0]['accuracy_score'])

@app.get("/select_estimator")
async def select_estimator_api(selected_estimator_id: int = 2):
    estimator_list = joblib.load('ml_pipeline/temporary_objects/estimator_list')
    selected_estimator= [estimator_list[selected_estimator_id]]
    joblib.dump(selected_estimator, 'ml_pipeline/temporary_objects/selected_estimator')
    return "estimator selected: "+ str(selected_estimator[0].__class__.__name__)

@app.get("/tune_estimator")
async def tune_estimator_api(n_iter: int= 10):

    selected_estimator = joblib.load('ml_pipeline/temporary_objects/selected_estimator')
    series =joblib.load('ml_pipeline/temporary_objects/XY')
    X, Y, test_X, test_Y= series['X'], series['Y'], series['test_X'], series['test_Y']
    tuned_estimator= local_functions.tune_estimators(X, Y, selected_estimator, n_iter= n_iter)
    joblib.dump(tuned_estimator, 'ml_pipeline/temporary_objects/tuned_estimator')
    metrics_list= local_functions.evaluate_estimators(test_X, test_Y, estimator_list= tuned_estimator)
    joblib.dump(metrics_list, 'ml_pipeline/temporary_objects/metrics_list' )

    return "Following hypermarameters are tuned: "+ str(tuned_estimator[0].get_params()) +"\t"+" performance of the model is: \t"+"accuracy_score: "+str(metrics_list[0]['accuracy_score'])








