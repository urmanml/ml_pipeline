import joblib
series = joblib.load('temporary_objects/XY')
X, Y = series['X'], series['Y']

dataset= X
X.select_dtypes(include='number')
x= dataset['Pregnancies']
import pandas as pd
dataset['Age_band']= pd.cut(dataset['Age'],bins= [0,25,45,100])
dataset['Pregnancies']= dataset['Pregnancies'].astype('object')
x= dataset['Pregnancies']

#X.select_dtypes(include=['category', 'object'])

def describe_numeric(dataset):
    dataset= dataset.select_dtypes(include='number')
    result= dataset.describe()
    result
    f= lambda x: int(x.isna().sum())

    result.loc['count_na',:]= dataset.apply(func=f)
    result.loc['perc_na',:]= result.loc['count_na',:]/result.loc['count',:]*100
    import scipy
    f= lambda x: scipy.stats.kurtosis(x)
    result.loc['kurtosis',:]= dataset.apply(func=f)

    def find_outliers(x):
        upper = x.quantile(.75)
        lower= x.quantile(.25)
        range= upper-lower
        upper_threshold= upper+1.5*range
        lower_threshold= lower-1.5*range

        outliers_upper= x[x>upper_threshold].to_list()
        outliers_lower= x[x<lower_threshold].to_list()
        outliers= outliers_upper+ outliers_lower
        return len(outliers)

    result.loc['n_outliers', :] = dataset.apply(func=find_outliers)

    return result




    return result

def describe_category(dataset):
    dataset= dataset.select_dtypes(include= ['category', 'object'])
    result= pd.DataFrame(columns= dataset.columns)

    f= lambda x: len(x)
    result.loc['count',:]= dataset.apply(func=f)
    f= lambda x: int(x.isna().sum())

    result.loc['count_na',:]= dataset.apply(func=f)
    result.loc['perc_na',:]= result.loc['count_na',:]/result.loc['count',:]*100

    f= lambda x: x.nunique()
    result.loc['count_distinct',:]= dataset.apply(func=f)
    f= lambda x: x.value_counts().index[0]
    result.loc['most_frequent_value',:]= dataset.apply(func=f)

    return result



def describe_dataframe(dataset):
    result_numeric= describe_numeric(dataset)
    result_category= describe_category(dataset)
    result= {'result_numeric':result_numeric, 'result_category':result_category}
    return result

describe_dataframe(dataset)

