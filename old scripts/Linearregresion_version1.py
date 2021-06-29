# importing the required libraries and packages
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

#Defining linear regression function which accepts the inputs and genarates the output
def LinearRegression1_def(DatasetLocation, OutputLocation, Target, columnNames, inputDatsetDelimiter, resultDelimiter):
    #reding the data
    df = pd.read_csv(DatasetLocation, delimiter=f"{inputDatsetDelimiter}", header='infer', index_col=[0])  # , low_memory=False)#encoding='unicode_escape')
    #saving the column names into cols list
    Cols = []
    for x in columnNames:
        # DriverDataset[x['driverColumn']] = DriverDataset[x['driverColumn']].astype(str)
        Cols.append(x['columnName'])

    #print("Cols")
    #print(target)
    #print(cols)
    #creating the indepedent and target variables
    X = df[Cols]
    Y = df[Target]
    #Spliting the data into train and test sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=5)
    #Running the Linear Regression Model
    model = LinearRegression()
    #Fitting the model
    model.fit(X_train, Y_train)
    ## model evaluation for training set
    y_train_predict = model.predict(X_train)
    rmse = (np.sqrt(mean_squared_error(Y_train, y_train_predict)))
    r2 = r2_score(Y_train, y_train_predict)
    #Metrix for training data
    print("The model performance for training set")
    print("--------------------------------------")
    print('RMSE is {}'.format(rmse))
    print('R2 score is {}'.format(r2))
    print("\n")
    #
    # model evaluation for testing set
    y_test_predict = model.predict(X_test)
    rmse = (np.sqrt(mean_squared_error(Y_test, y_test_predict)))
    r2 = r2_score(Y_test, y_test_predict)
    #Metrix for test data
    print("The model performance for testing set")
    print("--------------------------------------")
    print('RMSE is {}'.format(rmse))
    print('R2 score is {}'.format(r2))
    #Creating predictions columns
    predictions = pd.DataFrame(y_test_predict, columns=['Prediction'])
    #Merging the predictions to the orginal datset
    df_out = pd.merge(df, predictions, how='left', left_index=True, right_index=True)
    #Saving the data in given location
    df_out.to_csv(OutputLocation, sep=f'{resultDelimiter}')
    return "success"

