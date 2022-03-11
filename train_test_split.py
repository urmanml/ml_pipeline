
# import joblib
#
# dataset = joblib.load('temporary_objects/df')
# selected_columns = list(dataset.columns[0:-1])
# target='Outcome'
# create_validation_set= True
def split_data_X_Y(dataset, target, selected_columns ,test_size= .3,  seed= 1234 , create_validation_set= False ):
    import pandas as pd
    test_size= test_size
    import random
    from sklearn.model_selection import train_test_split
    # subset only selected columns

    dataset= dataset[selected_columns+[target]]

    X, test_X,Y,  test_Y = train_test_split(dataset.drop([target], axis= 1), dataset[[target]], test_size=test_size, random_state=seed)

    ## split test set further into test and validation set
    if create_validation_set== True:
        val_split= .5 ## percentage of test set to be kept aside for validation set
        test_X, val_X, test_Y,  val_Y = train_test_split(test_X, test_Y, test_size=.5,
                                                random_state=seed)

    series= pd.Series(dtype= 'object')
    series['X']=X
    series['Y'] = Y
    series['test_X'] = test_X
    series['test_Y'] = test_Y
    if create_validation_set== True:
        series['val_X'] = val_X
        series['val_Y'] = val_Y



    return series



