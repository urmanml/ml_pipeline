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
async def train_test_split_api(target='Outcome', test_ratio= .3, selected_columns=[]):
    """
    train test data split
    :param target: target variable name
    :param test_ratio: test ratio to be maintained
    :return:
    """
    print("websocket message: train test split started")

    dataset = joblib.load('temporary_objects/df')
    selected_columns= list(dataset.columns[0:-1])


    series, message= backend_functions.split_data_X_Y(dataset, target, selected_columns= selected_columns, test_size= .3)

    ### reconstruct train and test set for storing to delta lake
    train_set= pd.concat([series['X'], series['Y']], axis=1)
    test_set= pd.concat([series['test_X'], series['test_Y']], axis=1)

    # Store intermediate objects to specified location
    joblib.dump(series, 'temporary_objects/XY')

    # Store train and test set to delta lake for preview
    ## train_set.to_delta
    ## train_set.to_delta



    ## websocket message here to signal completion
    print("websocket message: train test split started")

    return message