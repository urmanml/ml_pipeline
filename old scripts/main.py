#importing required libraries and packages

1+1

import uvicorn
import fastapi
from typing import List, Optional
from fastapi import FastAPI, Query
from pydantic import BaseModel

#importing Models as definitions from the files
from Linearregresion_version1 import LinearRegression1_def

#assigning FastAPI as app
app = FastAPI()

### Linear Regression RestAPI
#Defining Base Model
#defining the inputs
class LinearRegressionRequest(BaseModel):
    #dataset location
    DatasetLocation: str


    #ouput file storage location
    OutputLocation: str
    #Target column
    Target: str
    #predictor columns
    columnNames: List[dict]
    #accepting delimiters as inputs
    inputDatsetDelimiter : str
    resultDelimiter : str

#defining the endpoint for LinearRegression
@app.post("/LinearRegressionRestAPI")
#calling the functions with given inputs
async def LinearRegressionRestAPI(LR1:LinearRegressionRequest):
    Result = LinearRegression1_def(LR1.DatasetLocation,LR1.OutputLocation,LR1.Target, LR1.columnNames, LR1.inputDatsetDelimiter, LR1.resultDelimiter)
    # ...
    # Runs the LinearRegression
    # ...
    return {f"{Result}"}
