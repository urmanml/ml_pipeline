def adder(num1: float= 2, num2: float= 3):
    print('hello')
    return num1+num2

import joblib
def split_data_X_Y(dataset, target, selected_columns ,test_size= .3):
    import pandas as pd
    seed= 1234
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

